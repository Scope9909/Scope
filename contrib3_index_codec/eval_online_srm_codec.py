"""Validation-select an online SRM state codec, then evaluate fixed Test once.

This is an online-state adaptation, not the Teacher T/E/fd/hd/B cache codec.
The dominant Train observation array is stored losslessly as uint16.  The
Train-fitted quantization scale is tested as float16 on Validation; it is kept
only if the raw Validation MAE increase is at most 0.01.
"""

import argparse
import importlib.util
import json
import logging
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import container

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=("PEMS03", "PEMS04", "PEMS07", "PEMS08"))
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--q8-pilot", action="store_true")
parser.add_argument("--qbits", type=int, choices=(8, 10, 12), default=8)
parser.add_argument("--nodewise", action="store_true")
parser.add_argument("--fixed-nodes", type=int, nargs="+", default=None,
                    help="Exploratory node set fixed before this Validation/Test run")
parser.add_argument("--size-scan", action="store_true")
parser.add_argument("--pack-index", action="store_true")
parser.add_argument("--validation-only", action="store_true")
parser.add_argument("--gate", type=float, default=0.01)
parser.add_argument("--temporal-audit", action="store_true")
cli = parser.parse_args()

# Reuse the frozen SRM model, retrieval, metric and calibration functions.
runner_path = ROOT / "srm_test_once" / "run_frozen_srm_test_once.py"
spec = importlib.util.spec_from_file_location("frozen_srm_runner", runner_path)
runner = importlib.util.module_from_spec(spec)
sys.argv = [str(runner_path), "--dataset", cli.dataset,
            "--seed", str(cli.seed), "--mode", "sparse"]
spec.loader.exec_module(runner)
sys.argv = [__file__]


def make_state(source, destination, scale_f16):
    with np.load(source, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    raw = np.asarray(state["train_raw"])
    if not (np.isfinite(raw).all() and np.array_equal(raw, np.rint(raw))
            and raw.min() >= 0 and raw.max() <= 65535):
        raise ValueError(f"Cannot losslessly encode train_raw: {source}")
    state["train_raw"] = raw.astype(np.uint16)
    if scale_f16:
        state["quant_scale"] = state["quant_scale"].astype(np.float16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **state)
    with np.load(destination, allow_pickle=False) as archive:
        if not np.array_equal(archive["train_raw"].astype(np.float32), raw):
            raise ValueError("Encoded Train observations failed roundtrip")
    return destination.stat().st_size


def make_q8_state(source, destination, bits=8):
    """Quantize immutable history; keep cold-start overlap frames exact."""
    with np.load(source, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    raw = np.asarray(state.pop("train_raw"), dtype=np.float32)
    cold = "cold_start_endpoint" in state
    exact_tail = int(np.asarray(state["input_length"]).item()) if cold else 0
    prefix = raw[:-exact_tail] if exact_tail else raw
    tail = raw[-exact_tail:] if exact_tail else raw[:0]
    top = prefix.max(axis=0)
    levels = (1 << bits) - 1
    scale = np.where(top > 0, top / float(levels), 1.0).astype(np.float32)
    codes = np.clip(np.rint(prefix / scale[None]), 0, levels).astype(np.uint16)
    state["train_raw_q8"] = (codes.astype(np.uint8) if bits == 8 else
                             container.pack_bits(codes, bits))
    state["train_raw_bits"] = np.asarray(bits, dtype=np.uint8)
    state["train_raw_prefix_shape"] = np.asarray(prefix.shape, dtype=np.int32)
    state["train_raw_scale"] = scale
    state["train_raw_tail"] = tail.astype(np.uint16)
    state["quant_scale"] = state["quant_scale"].astype(np.float16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **state)
    return destination.stat().st_size


def make_mixed_state(source, destination, q_nodes, pack_index=False):
    """Keep exact-history values for other nodes, Q8 only selected nodes."""
    with np.load(source, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    raw = np.asarray(state.pop("train_raw"), dtype=np.float32)
    nodes = raw.shape[1]
    q_nodes = np.asarray(sorted(set(q_nodes)), dtype=np.int32)
    if np.any(q_nodes < 0) or np.any(q_nodes >= nodes):
        raise ValueError("Q8 node index out of range")
    exact_nodes = np.setdiff1d(np.arange(nodes, dtype=np.int32), q_nodes)
    tail_len = int(np.asarray(state["input_length"]).item()) if "cold_start_endpoint" in state else 0
    prefix = raw[:-tail_len] if tail_len else raw
    tail = raw[-tail_len:] if tail_len else raw[:0]
    selected = prefix[:, q_nodes]
    top = selected.max(axis=0) if len(q_nodes) else np.empty(0, dtype=np.float32)
    scale = np.where(top > 0, top / 255.0, 1.0).astype(np.float32)
    state["train_raw_mixed_shape"] = np.asarray(prefix.shape, dtype=np.int32)
    state["train_raw_mixed_q_nodes"] = q_nodes
    state["train_raw_mixed_exact_nodes"] = exact_nodes
    state["train_raw_mixed_q_codes"] = np.clip(
        np.rint(selected / scale[None]), 0, 255).astype(np.uint8)
    state["train_raw_mixed_q_scale"] = scale
    state["train_raw_mixed_exact"] = prefix[:, exact_nodes].astype(np.uint16)
    state["train_raw_mixed_tail"] = tail.astype(np.uint16)
    state["quant_scale"] = state["quant_scale"].astype(np.float16)
    if pack_index:
        idx = np.asarray(state["idx_train"])
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
            raise ValueError("Cannot pack noncontiguous candidate indices")
        state.pop("idx_train")
        state["idx_train_start"] = np.asarray(idx[0], dtype=np.int32)
        state["idx_train_count"] = np.asarray(len(idx), dtype=np.int32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **state)
    return destination.stat().st_size


def load_memory(path):
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    if "idx_train_start" in state:
        start = int(np.asarray(state.pop("idx_train_start")).item())
        count = int(np.asarray(state.pop("idx_train_count")).item())
        state["idx_train"] = np.arange(start, start + count, dtype=np.int32)
    if "train_raw_q8" in state:
        codes = state.pop("train_raw_q8")
        bits = int(np.asarray(state.pop("train_raw_bits")).item())
        shape = tuple(state.pop("train_raw_prefix_shape"))
        scale = state.pop("train_raw_scale")
        tail = state.pop("train_raw_tail")
        if bits != 8:
            codes = container.unpack_bits(codes, int(np.prod(shape)), bits).reshape(shape)
        state["train_raw"] = np.concatenate(
            (codes.astype(np.float32) * scale[None], tail.astype(np.float32)), axis=0)
    if "train_raw_mixed_shape" in state:
        shape = tuple(state.pop("train_raw_mixed_shape"))
        q_nodes = state.pop("train_raw_mixed_q_nodes")
        exact_nodes = state.pop("train_raw_mixed_exact_nodes")
        codes = state.pop("train_raw_mixed_q_codes")
        scale = state.pop("train_raw_mixed_q_scale")
        exact = state.pop("train_raw_mixed_exact")
        tail = state.pop("train_raw_mixed_tail")
        prefix = np.empty(shape, dtype=np.float32)
        prefix[:, q_nodes] = codes.astype(np.float32) * scale[None]
        prefix[:, exact_nodes] = exact.astype(np.float32)
        state["train_raw"] = np.concatenate((prefix, tail.astype(np.float32)), axis=0)
    return runner.OnlineResidualMemory(state)


def evaluate():
    exp_dir, calibration_path, val_path, test_path, selector = runner.inputs()
    with (exp_dir / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.pearson_adj = None
    if not (args.use_analog_lowrank_retrieval_residual and
            args.use_analog_exact_match_residual):
        raise ValueError("Expected Student low-rank plus exact-history checkpoint")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("online_srm_codec")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = runner.datasets_utils.load_dataset(args, logger)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(
        exp_dir / "best.pth", map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False

    index_path = (ROOT / "datasets" / cli.dataset /
                  f"{cli.dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz")
    with np.load(index_path, allow_pickle=False) as index:
        val_endpoints = np.asarray(index["idx_val"], dtype=np.int64)
        test_endpoints = np.asarray(index["idx_test"], dtype=np.int64)
    output_dir = HERE / "packages" / "online"
    prefix = f"{cli.dataset.lower()}_s{cli.seed}"
    val_u16 = output_dir / f"{prefix}_val_u16.npz"
    val_f16 = output_dir / f"{prefix}_val_u16_scalef16.npz"
    test_u16 = output_dir / f"{prefix}_test_u16.npz"
    test_f16 = output_dir / f"{prefix}_test_u16_scalef16.npz"
    sizes = {
        "val_u16": make_state(val_path, val_u16, False),
        "val_u16_scalef16": make_state(val_path, val_f16, True),
        "test_u16": make_state(test_path, test_u16, False),
        "test_u16_scalef16": make_state(test_path, test_f16, True),
    }

    def collect_val(path):
        memory = load_memory(path)
        if memory.cold_start_endpoint != int(val_endpoints[0]):
            raise ValueError("Validation cold-start alignment mismatch")
        prediction, truth = runner.collect(
            loaders["val_loader"], val_endpoints, model, scaler, memory, device, 32)
        return prediction, truth, runner.metrics(prediction, truth)["mae"]

    raw_val_prediction, val_truth, raw_val_mae = collect_val(val_path)
    # The only change in val_u16 is an exact integer dtype conversion, which
    # make_state checks elementwise after reading the written archive.  The
    # OnlineResidualMemory constructor casts it back to float32.
    exact_val_prediction, exact_val_mae = raw_val_prediction, raw_val_mae
    f16_val_prediction, _, f16_val_mae = collect_val(val_f16)
    f16_delta = f16_val_mae - raw_val_mae
    use_f16 = bool(f16_delta <= 0.01)
    selected_val_prediction = f16_val_prediction if use_f16 else exact_val_prediction
    selected_test_path = test_f16 if use_f16 else test_u16
    if cli.fixed_nodes is not None:
        suffix = f"fixed_{'_'.join(map(str, cli.fixed_nodes))}{'_idx' if cli.pack_index else ''}"
        val_fixed = output_dir / f"{prefix}_val_{suffix}.npz"
        test_fixed = output_dir / f"{prefix}_test_{suffix}.npz"
        make_mixed_state(val_path, val_fixed, cli.fixed_nodes, pack_index=cli.pack_index)
        fixed_prediction, _, fixed_mae = collect_val(val_fixed)
        fixed_delta = fixed_mae - raw_val_mae
        temporal = None
        if cli.temporal_audit:
            split = int(len(val_truth) * 0.6)
            if split <= 0 or split >= len(val_truth):
                raise ValueError("Validation split has an empty block")
            temporal = {}
            for name, sl in (("selection_early_60pct", slice(None, split)),
                             ("confirmation_late_40pct", slice(split, None))):
                base_mae = runner.metrics(raw_val_prediction[sl], val_truth[sl])["mae"]
                coded_mae = runner.metrics(fixed_prediction[sl], val_truth[sl])["mae"]
                temporal[name] = dict(samples=len(val_truth[sl]),
                                      baseline_mae=base_mae, coded_mae=coded_mae,
                                      delta=coded_mae-base_mae,
                                      passed=coded_mae-base_mae <= cli.gate)
        zero_val = output_dir / f"{prefix}_val_fixed_zero{'_idx' if cli.pack_index else ''}.npz"
        zero_test = output_dir / f"{prefix}_test_fixed_zero{'_idx' if cli.pack_index else ''}.npz"
        make_mixed_state(val_path, zero_val, [], pack_index=cli.pack_index)
        make_mixed_state(test_path, zero_test, [], pack_index=cli.pack_index)
        if fixed_delta > cli.gate or val_fixed.stat().st_size >= zero_val.stat().st_size:
            result = dict(dataset=cli.dataset, seed=cli.seed,
                          fixed_nodes=cli.fixed_nodes, val_delta=fixed_delta,
                          val_bytes=val_fixed.stat().st_size,
                          same_layout_lossless_val_bytes=zero_val.stat().st_size,
                          temporal_audit=temporal,
                          gate=cli.gate, passed=False, test_evaluated=False)
            print(json.dumps(result, indent=2), flush=True)
            return result
        if cli.validation_only:
            result = dict(dataset=cli.dataset, seed=cli.seed,
                          fixed_nodes=cli.fixed_nodes, val_delta=fixed_delta,
                          val_bytes=val_fixed.stat().st_size,
                          same_layout_lossless_val_bytes=zero_val.stat().st_size,
                          temporal_audit=temporal,
                          gate=cli.gate, passed=True, test_evaluated=False)
            print(json.dumps(result, indent=2), flush=True)
            return result
        make_mixed_state(test_path, test_fixed, cli.fixed_nodes, pack_index=cli.pack_index)
        sizes["same_layout_lossless_test"] = zero_test.stat().st_size
        sizes["same_layout_lossless_val"] = zero_val.stat().st_size
        selected_val_prediction = fixed_prediction
        selected_test_path = test_fixed
    if cli.nodewise:
        node_count = runner.OnlineResidualMemory.from_npz(val_path).train_raw.shape[1]
        trial_dir = output_dir / f"{prefix}_node_trials"
        rows = []
        candidates = []
        for node in range(node_count):
            trial = trial_dir / f"node_{node:03d}.npz"
            size = make_mixed_state(val_path, trial, [node])
            if size >= sizes["val_u16_scalef16"]:
                rows.append(dict(node=node, bytes=size, screened="not_smaller"))
                continue
            _, _, mae = collect_val(trial)
            delta = mae - raw_val_mae
            rows.append(dict(node=node, bytes=size, val_delta=delta))
            if delta <= 0.01:
                candidates.append(node)
            print(f"node {node}/{node_count} bytes={size} delta={delta:+.6f}", flush=True)
        chosen = candidates.copy()
        joint_path = output_dir / f"{prefix}_val_nodewise_joint.npz"
        joint_size = make_mixed_state(val_path, joint_path, chosen)
        joint_prediction, _, joint_mae = collect_val(joint_path)
        joint_delta = joint_mae - raw_val_mae
        order = sorted(candidates, key=lambda n: next(
            row["val_delta"] for row in rows if row["node"] == n), reverse=True)
        restored = []
        for node in order:
            if joint_delta <= 0.01 and joint_size < sizes["val_u16_scalef16"]:
                break
            chosen.remove(node)
            restored.append(node)
            joint_size = make_mixed_state(val_path, joint_path, chosen)
            joint_prediction, _, joint_mae = collect_val(joint_path)
            joint_delta = joint_mae - raw_val_mae
            print(f"restore node {node}, joint delta={joint_delta:+.6f}", flush=True)
        report = dict(dataset=cli.dataset, seed=cli.seed,
                      baseline_val_mae=raw_val_mae, gate=0.01,
                      per_node=rows, chosen_nodes=chosen, restored_nodes=restored,
                      joint_val_mae=joint_mae, joint_val_delta=joint_delta,
                      val_baseline_bytes=sizes["val_u16_scalef16"],
                      val_joint_bytes=joint_size,
                      test_evaluated=False)
        report_path = output_dir / f"{prefix}_nodewise_selection.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "per_node"},
                         indent=2), flush=True)
        if not chosen or joint_delta > 0.01 or joint_size >= sizes["val_u16_scalef16"]:
            return report
        test_joint = output_dir / f"{prefix}_test_nodewise_joint.npz"
        make_mixed_state(test_path, test_joint, chosen)
        selected_val_prediction = joint_prediction
        selected_test_path = test_joint
    q8_val_mae = None
    q8_val_delta = None
    if cli.q8_pilot:
        val_q8 = output_dir / f"{prefix}_val_historyq{cli.qbits}.npz"
        sizes["val_historyq8"] = make_q8_state(val_path, val_q8, cli.qbits)
        q8_val_prediction, _, q8_val_mae = collect_val(val_q8)
        q8_val_delta = q8_val_mae - raw_val_mae
        if q8_val_delta > cli.gate or cli.validation_only:
            result = dict(dataset=cli.dataset, seed=cli.seed,
                          q8_val_mae=q8_val_mae, q8_val_delta=q8_val_delta,
                          gate=cli.gate, passed=bool(q8_val_delta <= cli.gate),
                          baseline_val_mae=raw_val_mae,
                          baseline_u16_f16_bytes=sizes["val_u16_scalef16"],
                          q8_bytes=sizes["val_historyq8"],
                          test_evaluated=False)
            print(json.dumps(result, indent=2), flush=True)
            return result
        test_q8 = output_dir / f"{prefix}_test_historyq{cli.qbits}.npz"
        sizes["test_historyq8"] = make_q8_state(test_path, test_q8, cli.qbits)
        selected_val_prediction = q8_val_prediction
        selected_test_path = test_q8
    test_memory = load_memory(selected_test_path)
    if test_memory.cold_start_endpoint is not None:
        raise ValueError("Test state is not fully observed Train state")
    test_prediction, test_truth = runner.collect(
        loaders["test_loader"], test_endpoints, model, scaler, test_memory, device, 32)
    parameters = json.loads(calibration_path.read_text(encoding="utf-8"))
    kwargs = {key: parameters[key] for key in (
        "num_bins", "smooth_shrinkage", "num_time_bins",
        "time_shrinkage", "slices_per_day")}
    _, calibrated, _ = runner.fit_and_apply(
        selected_val_prediction, val_truth, val_endpoints,
        test_prediction, test_endpoints, **kwargs)
    result = {
        "dataset": cli.dataset, "seed": cli.seed,
        "scope": "Student+ValCal+SRM online state, not Teacher cache",
        "selector": selector, "gate": cli.gate,
        "val_raw_mae": raw_val_mae,
        "val_u16_mae": exact_val_mae,
        "val_u16_scalef16_mae": f16_val_mae,
        "val_scalef16_delta": f16_delta,
        "val_fixed_mae": fixed_mae if cli.fixed_nodes is not None else None,
        "val_fixed_delta": fixed_delta if cli.fixed_nodes is not None else None,
        "selected": "nodewise_joint" if cli.nodewise else
                    suffix if cli.fixed_nodes is not None else
                    f"historyq{cli.qbits}" if cli.q8_pilot else
                    "u16_scalef16" if use_f16 else "u16",
        "q8_val_mae": q8_val_mae,
        "q8_val_delta": q8_val_delta,
        "test_raw_mae": runner.metrics(test_prediction, test_truth)["mae"],
        "test_calibrated_mae": runner.metrics(calibrated, test_truth)["mae"],
        "test_state_original_bytes": test_path.stat().st_size,
        "test_state_encoded_bytes": selected_test_path.stat().st_size,
        "test_state_saving_fraction": 1 - selected_test_path.stat().st_size / test_path.stat().st_size,
        "sizes": sizes,
        "encoded_test_state": str(selected_test_path),
    }
    report_tag = ('nodewise_joint' if cli.nodewise else
                  suffix if cli.fixed_nodes is not None else
                  f"historyq{cli.qbits}" if cli.q8_pilot else 'online_codec')
    report_path = output_dir / f"{prefix}_{report_tag}_eval.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    if cli.size_scan:
        _, _, val_path, test_path, _ = runner.inputs()
        output_dir = HERE / "packages" / "online"
        with tempfile.TemporaryDirectory(dir=output_dir) as directory:
            trial = Path(directory) / "trial.npz"
            baseline = make_mixed_state(val_path, trial, [], pack_index=True)
            with np.load(val_path, allow_pickle=False) as archive:
                nodes = archive["train_raw"].shape[1]
            candidates = []
            for node in range(nodes):
                size = make_mixed_state(val_path, trial, [node], pack_index=True)
                if size < baseline:
                    candidates.append(dict(node=node, val_bytes=size, saving=baseline-size))
            print(json.dumps(dict(dataset=cli.dataset, seed=cli.seed,
                                  baseline_val_bytes=baseline,
                                  smaller_single_nodes=candidates), indent=2))
    else:
        evaluate()
