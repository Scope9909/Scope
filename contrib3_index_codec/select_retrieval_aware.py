"""Exploratory Validation-only comparison of online-state node selection rules.

Both rules use the existing mixed Q8 codec and the same candidate pool.  The
retrieval-aware rule ranks candidates using changes in the *retrieval output*
and exact-hit availability on a deterministic Validation probe.  This is not
an audit of top-k candidate identities and does not guarantee hit preservation.
Test metrics and labels are never used for selection; dataset initialization may
still construct a Test loader, as required by the existing project API.
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
parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=("PEMS03", "PEMS04", "PEMS07", "PEMS08"))
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gate", type=float, default=0.02)
parser.add_argument("--topk-gate", type=float,
                    help="Optional maximum full-probe top-k set flip rate")
parser.add_argument("--exact-gate", type=float,
                    help="Optional maximum full-probe exact-availability flip rate")
parser.add_argument("--strong-lossless-bytes", type=int,
                    help="Optional true lossless baseline; final package must be smaller")
parser.add_argument("--max-candidates", type=int, default=8)
parser.add_argument("--candidate-nodes", type=int, nargs="+",
                    help="Optional fixed node pool for a stability audit")
parser.add_argument("--audit-single-nodes", action="store_true",
                    help="Run a full Validation pass for every single-node package")
parser.add_argument("--probe-batches", type=int, default=3)
parser.add_argument("--hit-weight", type=float, default=2.0)
parser.add_argument("--output", type=Path)
cli = parser.parse_args()
if cli.max_candidates < 1 or cli.probe_batches < 1 or cli.hit_weight < 0:
    parser.error("max-candidates and probe-batches must be positive; hit-weight nonnegative")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from analog_distance import masked_history_distance
sys.argv = [str(HERE / "eval_online_srm_codec.py"), cli.dataset, "--seed", str(cli.seed)]
spec = importlib.util.spec_from_file_location("online_codec_eval", HERE / "eval_online_srm_codec.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)
runner = codec.runner
sys.argv = [__file__]


def setup():
    experiment, _, val_path, _, _ = runner.inputs()
    with (experiment / "config.pkl").open("rb") as stream:
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
    loaders, nodes, scaler, _ = runner.datasets_utils.load_dataset(
        args, logging.getLogger("retrieval_aware_codec"))
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(
        experiment / "best.pth", map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False
    index_path = ROOT / "datasets" / cli.dataset / (
        f"{cli.dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz")
    with np.load(index_path, allow_pickle=False) as archive:
        val_endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
    return val_path, loaders["val_loader"], val_endpoints, scaler, model, device


def probe_inputs(loader, scaler, memory, batch_limit):
    """Use the first fixed Validation batches; query histories are never encoded."""
    old = loader.bs, loader.drop_last, loader.num_batch
    loader.bs = 32
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / 32))
    batches = []
    try:
        for number, (loader_input, _) in enumerate(loader.get_iterator()):
            if number >= batch_limit:
                break
            x, x_time, *_ = runner.unpack_dataloader_input(loader_input)
            history = np.rint(scaler.inverse_transform(x)[..., 0][
                :, :, memory.node_ids]).astype(np.float32)
            tod = np.asarray(x_time[:, -1, 0], dtype=np.int16)
            batches.append((history, tod))
    finally:
        loader.bs, loader.drop_last, loader.num_batch = old
    if not batches:
        raise ValueError("Empty Validation probe")
    return batches


def retrieval_change(reference, candidate, batches):
    hit_flips = 0
    retrieved_difference = 0.0
    retrieved_count = 0
    topk_sets = 0
    topk_set_flips = 0
    topk_overlap = 0
    top1_flips = 0
    base_margin_sum = 0.0
    base_margin_count = 0
    flipped_margin_sum = 0.0
    flipped_margin_count = 0
    for history, tod in batches:
        base = reference.query(history, tod)
        coded = candidate.query(history, tod)
        hit_flips += int(np.count_nonzero(base[2] != coded[2]))
        difference = np.abs(base[0].astype(np.float64) - coded[0].astype(np.float64))
        retrieved_difference += float(difference.sum())
        retrieved_count += difference.size
        for value in np.unique(tod):
            query_positions = np.flatnonzero(tod == value)
            train_positions = reference._train_positions_by_tod.get(int(value))
            candidate_positions = candidate._train_positions_by_tod.get(int(value))
            if (train_positions is None or candidate_positions is None or
                    not np.array_equal(reference.idx_train[train_positions],
                                       candidate.idx_train[candidate_positions])):
                raise ValueError("Reference/candidate retrieval pools are not aligned")
            endpoints = reference.idx_train[train_positions]
            base_history = reference.train_raw[
                endpoints[:, None] + reference.x_offsets[None, :]]
            coded_history = candidate.train_raw[
                endpoints[:, None] + candidate.x_offsets[None, :]]
            queries = history[query_positions]
            base_distance = masked_history_distance(
                queries, base_history, reference.min_valid_history,
                metric="masked_mse")
            coded_distance = masked_history_distance(
                queries, coded_history, candidate.min_valid_history,
                metric="masked_mse")
            k = reference.k
            if base_distance.shape[1] <= k:
                raise ValueError("Need at least k+1 candidates for a margin audit")
            base_nearest = np.argpartition(
                base_distance, kth=k - 1, axis=1)[:, :k, :].transpose(0, 2, 1)
            coded_nearest = np.argpartition(
                coded_distance, kth=k - 1, axis=1)[:, :k, :].transpose(0, 2, 1)
            base_sorted = np.sort(base_nearest, axis=2)
            coded_sorted = np.sort(coded_nearest, axis=2)
            flipped = np.any(base_sorted != coded_sorted, axis=2)
            overlap = (base_nearest[..., :, None] ==
                       coded_nearest[..., None, :]).any(axis=3).sum(axis=2)
            base_top1 = np.argmin(base_distance, axis=1)
            coded_top1 = np.argmin(coded_distance, axis=1)
            partitioned = np.partition(base_distance, kth=k, axis=1)
            lower = partitioned[:, k - 1, :]
            upper = partitioned[:, k, :]
            finite = np.isfinite(lower) & np.isfinite(upper)
            margin = np.zeros_like(lower, dtype=np.float32)
            np.subtract(upper, lower, out=margin, where=finite)
            topk_sets += int(flipped.size)
            topk_set_flips += int(flipped.sum())
            topk_overlap += int(overlap.sum())
            top1_flips += int(np.count_nonzero(base_top1 != coded_top1))
            base_margin_sum += float(margin[finite].sum())
            base_margin_count += int(finite.sum())
            flipped_finite = flipped & finite
            flipped_margin_sum += float(margin[flipped_finite].sum())
            flipped_margin_count += int(flipped_finite.sum())
    return dict(
        exact_hit_flips=hit_flips,
        exact_hit_flip_rate=hit_flips / retrieved_count,
        mean_abs_retrieval_change=retrieved_difference / retrieved_count,
        topk_set_count=topk_sets,
        topk_set_flips=topk_set_flips,
        topk_set_flip_rate=topk_set_flips / topk_sets,
        mean_topk_overlap=topk_overlap / topk_sets,
        top1_flips=top1_flips,
        top1_flip_rate=top1_flips / topk_sets,
        mean_base_k_boundary_margin=(base_margin_sum / base_margin_count
                                     if base_margin_count else None),
        mean_flipped_base_k_boundary_margin=(
            flipped_margin_sum / flipped_margin_count
            if flipped_margin_count else None),
    )


def main():
    val_path, loader, endpoints, scaler, model, device = setup()
    if cli.output is None:
        output = HERE / "packages" / "online" / (
            f"{cli.dataset.lower()}_s{cli.seed}_retrieval_aware_val.json")
    else:
        output = cli.output
    with tempfile.TemporaryDirectory(prefix="retrieval_aware_") as directory:
        temporary = Path(directory)
        base_path = temporary / "lossless.npz"
        base_bytes = codec.make_mixed_state(val_path, base_path, [], pack_index=True)
        base_memory = codec.load_memory(base_path)
        batches = probe_inputs(loader, scaler, base_memory, cli.probe_batches)
        node_count = base_memory.train_raw.shape[1]
        candidates = []
        for node in range(node_count):
            path = temporary / f"node_{node}.npz"
            size = codec.make_mixed_state(val_path, path, [node], pack_index=True)
            saving = base_bytes - size
            if saving > 0:
                candidates.append(dict(node=node, single_bytes=size,
                                       single_saving=saving))
        # Bound expensive retrieval probes and full-Validation passes equally
        # for both rules; the pool is selected from bytes alone.
        candidates.sort(key=lambda item: (-item["single_saving"], item["node"]))
        if cli.candidate_nodes is not None:
            requested = set(cli.candidate_nodes)
            candidates = [row for row in candidates if row["node"] in requested]
            missing = requested.difference(row["node"] for row in candidates)
            if missing:
                raise ValueError(f"Requested nodes do not save bytes: {sorted(missing)}")
        candidates = candidates[:cli.max_candidates]
        for row in candidates:
            path = temporary / f"node_{row['node']}.npz"
            row.update(retrieval_change(base_memory, codec.load_memory(path), batches))
            # Exact-hit flips are weighted separately; the retrieval term is
            # dimensionless after scaling by the number of probe outputs.
            row["risk"] = (row["mean_abs_retrieval_change"] +
                           cli.hit_weight * row["exact_hit_flips"] /
                           sum(len(h) * base_memory.predict_length * node_count
                               for h, _ in batches))
        prediction, truth = runner.collect(
            loader, endpoints, model, scaler, base_memory, device, 32)
        q0_mae = runner.metrics(prediction, truth)["mae"]
        # Match the existing experiment's MAE gate: compare against the raw
        # online state, not the mixed container's float16 quant_scale.
        raw_prediction, raw_truth = runner.collect(
            loader, endpoints, model, scaler,
            runner.OnlineResidualMemory.from_npz(val_path), device, 32)
        baseline_mae = runner.metrics(raw_prediction, raw_truth)["mae"]

        if cli.audit_single_nodes:
            for row in candidates:
                path = temporary / f"node_{row['node']}.npz"
                predicted, target = runner.collect(
                    loader, endpoints, model, scaler,
                    codec.load_memory(path), device, 32)
                row["single_val_delta"] = (
                    runner.metrics(predicted, target)["mae"] - baseline_mae)

        def select(name, order):
            chosen = []
            steps = []
            current_bytes = base_bytes
            for row in order:
                trial = sorted(chosen + [row["node"]])
                path = temporary / f"{name}_joint.npz"
                size = codec.make_mixed_state(val_path, path, trial, pack_index=True)
                if size >= current_bytes:
                    steps.append(dict(node=row["node"], accepted=False,
                                      reason="not_smaller", bytes=size))
                    continue
                memory = codec.load_memory(path)
                predicted, target = runner.collect(
                    loader, endpoints, model, scaler, memory, device, 32)
                delta = runner.metrics(predicted, target)["mae"] - baseline_mae
                stability = None
                accepted = bool(delta <= cli.gate)
                if cli.topk_gate is not None or cli.exact_gate is not None:
                    stability = retrieval_change(
                        codec.load_memory(base_path), codec.load_memory(path), batches)
                    if cli.topk_gate is not None:
                        accepted = (accepted and
                                    stability["topk_set_flip_rate"] <= cli.topk_gate)
                    if cli.exact_gate is not None:
                        accepted = (accepted and
                                    stability["exact_hit_flip_rate"] <= cli.exact_gate)
                steps.append(dict(node=row["node"], accepted=accepted,
                                  bytes=size, val_delta=delta,
                                  stability=stability))
                if accepted:
                    chosen, current_bytes = trial, size
            # Re-evaluate the final combination: rejected trials overwrite
            # the temporary joint file, and only the final result is reported.
            path = temporary / f"{name}_final.npz"
            final_bytes = codec.make_mixed_state(val_path, path, chosen, pack_index=True)
            final_memory = codec.load_memory(path)
            # collect() may causally release cold-start candidates, so never
            # reuse the baseline memory object after a full Validation pass.
            final_probe_change = retrieval_change(
                codec.load_memory(base_path), final_memory, batches)
            predicted, target = runner.collect(
                loader, endpoints, model, scaler, final_memory, device, 32)
            final_delta = runner.metrics(predicted, target)["mae"] - baseline_mae
            final_passed = final_delta <= cli.gate
            if cli.topk_gate is not None:
                final_passed = (final_passed and final_probe_change[
                    "topk_set_flip_rate"] <= cli.topk_gate)
            if cli.exact_gate is not None:
                final_passed = (final_passed and final_probe_change[
                    "exact_hit_flip_rate"] <= cli.exact_gate)
            deployment_viable = (cli.strong_lossless_bytes is None or
                                   final_bytes < cli.strong_lossless_bytes)
            final_passed = final_passed and deployment_viable
            return dict(nodes=chosen, bytes=final_bytes,
                        saving=base_bytes - final_bytes, val_delta=final_delta,
                        passed=final_passed,
                        deployment_viable=deployment_viable,
                        delta_vs_strong_lossless=(None if cli.strong_lossless_bytes is None
                                                  else final_bytes - cli.strong_lossless_bytes),
                        retrieval_probe=final_probe_change, steps=steps)

        cost_order = sorted(candidates, key=lambda r: (-r["single_saving"], r["node"]))
        aware_order = sorted(candidates, key=lambda r: (
            -r["single_saving"] / (1 + 1000 * r["risk"]), r["node"]))
        result = dict(dataset=cli.dataset, seed=cli.seed,
                      scope="Validation only; exploratory; no Test selection",
                      gate=cli.gate, max_candidates=cli.max_candidates,
                      topk_gate=cli.topk_gate, exact_gate=cli.exact_gate,
                      strong_lossless_bytes=cli.strong_lossless_bytes,
                      retrieval_k=base_memory.k,
                      probe_batches=cli.probe_batches, hit_weight=cli.hit_weight,
                      same_layout_q0_val_bytes=base_bytes,
                      same_layout_q0_val_mae=q0_mae,
                      raw_val_mae=baseline_mae,
                      candidates=candidates,
                      cost_only=select("cost", cost_order),
                      retrieval_aware=select("aware", aware_order))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "candidates"}, indent=2), flush=True)
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
