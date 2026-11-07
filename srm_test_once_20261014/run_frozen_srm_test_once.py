"""Frozen SRM system comparison: fit on Validation, evaluate Test once.

The three modes use the same checkpoint for each dataset/seed. Sparse nodes are
fixed from prior Validation work. This script does no Test-based selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CONFIGS = {
    "PEMS03": {
        "experiments": {
            1: "TLAST_improved_PEMS03_12to12_exp10101500_s1",
            2: "TLAST_improved_PEMS03_12to12_exp10101506_s2",
            3: "TLAST_improved_PEMS03_12to12_exp10101513_s3",
        },
        "calibration": "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}_valcal/calibration.json",
        "validation_state_root": "pems03_int8_uniform_20261014",
        "test_state_root": "pems03_int8_uniform_20261014",
        "validation_state_pattern": "{mode}_cold_state.npz",
        "test_state_pattern": "{mode}_online_state.npz",
        "selector": "first55_three_block_stability_int8_topk36_budget",
    },
    "PEMS04": {
        "experiment": "TLAST_improved_PEMS04_12to12_exppems04_joint3_lite_r256_strictalign_s{seed}",
        "calibration": "paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_s{seed}_valcal/calibration.json",
        "validation_state_root": "cold_start_streaming_20261014",
        "validation_state_pattern": "pems04_{mode}_cold_state.npz",
        "test_states": {
            "sparse": "paper_results/icde_retrieval_free_pilot_20261012/online_state_budget_pems04_s1_cap2905847_20261013/online_train_state.npz",
            "full": "online_end_to_end_20261014/pems04_full_online_state.npz",
        },
        "selector": "actual_online_state_byte_capped_int8_s1",
    },
    "PEMS07": {
        "experiment": "TLAST_improved_PEMS07_12to12_exppems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}",
        "calibration": "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}_valcal/calibration.json",
        "validation_state_root": "cold_start_streaming_20261014",
        "validation_state_pattern": "pems07_{mode}_cold_state.npz",
        "test_states": {
            "sparse": "paper_results/icde_retrieval_free_pilot_20261012/pems07_online_state_audit_s1_20261013/robust_train_state.npz",
            "full": "online_end_to_end_20261014/pems07_full_online_state.npz",
        },
        "selector": "first55_three_block_stability_int8_s1",
    },
    "PEMS08": {
        "experiment": "TLAST_improved_PEMS08_12to12_exppems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}",
        "calibration": "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}_valcal/calibration.json",
        "validation_state_root": "cold_start_streaming_20261014",
        "validation_state_pattern": "pems08_{mode}_cold_state.npz",
        "test_states": {
            "sparse": "paper_results/icde_retrieval_free_pilot_20261012/online_state_budget_pems08_s1_cap2785289_20261013/online_train_state.npz",
            "full": "online_end_to_end_20261014/pems08_full_online_state.npz",
        },
        "selector": "actual_online_state_byte_capped_int8_s1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--mode", choices=("backbone", "sparse", "full"), required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--freeze-manifest", action="store_true")
    return parser.parse_args()


CLI = parse_args()
sys.argv = [sys.argv[0]]
from run import trainer, unpack_dataloader_input  # noqa: E402
from utils import datasets_utils  # noqa: E402
from utils.online_residual_memory import OnlineResidualMemory  # noqa: E402
from calibrate_dualscale_quantile_residual import metrics  # noqa: E402
from calibrate_dualscale_smooth_tod import fit_and_apply  # noqa: E402


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def inputs() -> tuple[Path, Path, Path | None, Path | None, str]:
    config = CONFIGS[CLI.dataset]
    experiment = config.get("experiments", {}).get(CLI.seed)
    if experiment is None:
        experiment = config["experiment"].format(seed=CLI.seed)
    exp_dir = ROOT / "EXP_experiments" / "TLAST_improved" / CLI.dataset / experiment
    calibration = ROOT / config["calibration"].format(seed=CLI.seed)
    if CLI.mode == "backbone":
        val_state = test_state = None
    else:
        val_state = (ROOT / config["validation_state_root"] /
                     config["validation_state_pattern"].format(mode=CLI.mode))
        if "test_states" in config:
            test_state = ROOT / config["test_states"][CLI.mode]
        else:
            test_state = (ROOT / config["test_state_root"] /
                          config["test_state_pattern"].format(mode=CLI.mode))
    for path in (exp_dir / "best.pth", exp_dir / "config.pkl", calibration,
                 val_state, test_state):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    return exp_dir, calibration, val_state, test_state, config["selector"]


def sparse_correction(model, prediction, ids, retrieval, exact, available):
    low = model.analog_lowrank_retrieval_residual
    exact_adapter = model.analog_exact_match_residual
    selected = prediction.index_select(2, ids)
    logits = low.horizon_logits[:, None] + torch.einsum(
        "nr,hr->hn", low.node_factor.index_select(0, ids), low.horizon_factor)
    blend = low.max_blend * torch.tanh(logits).view(1, logits.shape[0], len(ids), 1)
    selected = selected + blend * (retrieval - selected)
    exact_blend = (exact_adapter.max_blend *
                   torch.tanh(exact_adapter.horizon_logits)).view(1, -1, 1, 1)
    selected = selected + exact_blend * available.to(selected.dtype) * (exact - selected)
    return prediction.index_copy(2, ids, selected)


def collect(loader, endpoints, model, scaler, memory, device, batch_size):
    old_drop, old_batches, old_bs = loader.drop_last, loader.num_batch, loader.bs
    loader.bs = batch_size
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    predictions, truths = [], []
    cursor = 0
    ids = (None if memory is None else
           torch.as_tensor(memory.node_ids, dtype=torch.long, device=device))
    try:
        with torch.inference_mode():
            for loader_input, loader_target in loader.get_iterator():
                if cursor >= len(endpoints):
                    break
                x, x_time, *extras = unpack_dataloader_input(loader_input)
                if any(value is not None for value in extras):
                    raise ValueError("Precomputed query cache unexpectedly loaded")
                y, y_time = loader_target
                take = min(len(x), len(endpoints) - cursor)
                x, x_time, y, y_time = x[:take], x_time[:take], y[:take], y_time[:take]
                if memory is not None:
                    raw_history = np.rint(
                        scaler.inverse_transform(x)[..., 0][:, :, memory.node_ids]
                    ).astype(np.float32)
                    tod = np.asarray(x_time[:, -1, 0], dtype=np.int16)
                    if memory.cold_start_endpoint is None:
                        retrieval, exact, available, _ = memory.query(raw_history, tod)
                    else:
                        retrieval, exact, available, _ = memory.query_stream(
                            raw_history, endpoints[cursor:cursor + take], tod)
                model_input = [
                    torch.as_tensor(x, dtype=torch.float32, device=device),
                    torch.as_tensor(x_time, dtype=torch.long, device=device),
                    torch.as_tensor(y_time, dtype=torch.long, device=device),
                ]
                prediction, _ = model(model_input)
                if memory is not None:
                    prediction = sparse_correction(
                        model, prediction, ids,
                        torch.as_tensor(retrieval, device=device),
                        torch.as_tensor(exact, device=device).float(),
                        torch.as_tensor(available, device=device))
                predictions.append(scaler.inverse_transform(prediction).cpu().numpy()[..., 0])
                truths.append(np.asarray(y)[..., 0])
                cursor += take
    finally:
        loader.drop_last, loader.num_batch, loader.bs = old_drop, old_batches, old_bs
    if cursor != len(endpoints):
        raise ValueError("Loader did not cover the official split")
    return np.concatenate(predictions), np.concatenate(truths)


def main() -> None:
    if CLI.freeze_manifest:
        manifest_path = OUT / "frozen_manifest.json"
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        original = (CLI.dataset, CLI.seed, CLI.mode)
        jobs = []
        for dataset in CONFIGS:
            for seed in (1, 2, 3):
                for mode in ("backbone", "sparse", "full"):
                    CLI.dataset, CLI.seed, CLI.mode = dataset, seed, mode
                    exp_dir, calibration, val_state, test_state, selector = inputs()
                    jobs.append({
                        "dataset": dataset, "seed": seed, "mode": mode,
                        "selector": selector,
                        "checkpoint": str(exp_dir / "best.pth"),
                        "checkpoint_sha256": digest(exp_dir / "best.pth"),
                        "calibration": str(calibration),
                        "calibration_sha256": digest(calibration),
                        "validation_state": str(val_state) if val_state else None,
                        "validation_state_sha256": digest(val_state) if val_state else None,
                        "test_state": str(test_state) if test_state else None,
                        "test_state_sha256": digest(test_state) if test_state else None,
                    })
        CLI.dataset, CLI.seed, CLI.mode = original
        manifest_path.write_text(json.dumps({
            "protocol": "frozen_srm_system_test_once_v1",
            "purpose": "Evaluate existing per-dataset SRM configurations, not a unified selector",
            "test_evaluated_at_freeze": False,
            "test_based_selection": False,
            "expected_jobs": 36,
            "batch_size_for_accuracy": 32,
            "test_passes_per_job": 1,
            "calibration_fit_split": "validation_only",
            "runner_sha256": digest(Path(__file__)),
            "jobs": jobs,
        }, indent=2), encoding="utf-8")
        print(f"Frozen {len(jobs)} jobs: {manifest_path}")
        return
    exp_dir, calibration_path, val_path, test_path, selector = inputs()
    output = OUT / f"{CLI.dataset.lower()}_s{CLI.seed}_{CLI.mode}_test_once.json"
    if output.exists():
        raise FileExistsError(f"Test-once output already exists: {output}")
    if CLI.preflight:
        print(json.dumps({"dataset": CLI.dataset, "seed": CLI.seed,
                          "mode": CLI.mode, "checkpoint": str(exp_dir / "best.pth"),
                          "calibration": str(calibration_path),
                          "validation_state": str(val_path) if val_path else None,
                          "test_state": str(test_path) if test_path else None,
                          "selector": selector}, indent=2))
        return

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
        raise ValueError("Expected frozen low-rank plus exact-history checkpoint")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("srm_test_once")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(
        exp_dir / "best.pth", map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False

    index_path = (ROOT / "datasets" / CLI.dataset /
                  f"{CLI.dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz")
    with np.load(index_path, allow_pickle=False) as index:
        val_endpoints = np.asarray(index["idx_val"], dtype=np.int64)
        test_endpoints = np.asarray(index["idx_test"], dtype=np.int64)
    with np.load(ROOT / "datasets" / CLI.dataset /
                 f"{CLI.dataset}_time.npz", allow_pickle=False) as archive:
        time_data = np.asarray(archive["time"])
    val_memory = OnlineResidualMemory.from_npz(val_path) if val_path else None
    test_memory = OnlineResidualMemory.from_npz(test_path) if test_path else None
    if val_memory is not None:
        if val_memory.retrieval_format != "int8" or test_memory.retrieval_format != "int8":
            raise ValueError("Frozen comparison requires INT8 memory")
        if not np.array_equal(val_memory.node_ids, test_memory.node_ids):
            raise ValueError("Validation and Test nodes differ")
        if val_memory.cold_start_endpoint != int(val_endpoints[0]):
            raise ValueError("Validation cold-start state is misaligned")
        if test_memory.cold_start_endpoint is not None:
            raise ValueError("Test requires fully observed Train-candidate state")
        if CLI.mode == "full" and not np.array_equal(
                val_memory.node_ids, np.arange(nodes)):
            raise ValueError("Full state is not all-node")
        if CLI.mode == "sparse" and len(val_memory.node_ids) >= nodes:
            raise ValueError("Sparse state is not a proper subset")
        val_memory.validate_causality(
            val_endpoints, np.asarray(time_data[val_endpoints, 0], dtype=np.int16))
        test_memory.validate_causality(
            test_endpoints, np.asarray(time_data[test_endpoints, 0], dtype=np.int16))

    # All fitting uses Validation. No Test target is read before this point.
    val_prediction, val_truth = collect(
        loaders["val_loader"], val_endpoints, model, scaler, val_memory, device, 32)
    parameters = json.loads(calibration_path.read_text(encoding="utf-8"))
    kwargs = {key: parameters[key] for key in (
        "num_bins", "smooth_shrinkage", "num_time_bins",
        "time_shrinkage", "slices_per_day")}

    # The Test split is forwarded exactly once under the already frozen plan.
    test_prediction, test_truth = collect(
        loaders["test_loader"], test_endpoints, model, scaler, test_memory, device, 32)
    _, calibrated, _ = fit_and_apply(
        val_prediction, val_truth, val_endpoints,
        test_prediction, test_endpoints, **kwargs)
    report = {
        "protocol": "frozen_srm_online_test_once_v1",
        "dataset": CLI.dataset, "seed_label": CLI.seed, "mode": CLI.mode,
        "selector": selector, "test_evaluated": True,
        "test_evaluations": 1, "test_based_selection": False,
        "query_cache_loaded": False, "memory_format": "int8" if test_memory else None,
        "batch_size": 32, "validation_samples": len(val_endpoints),
        "test_samples": len(test_endpoints),
        "checkpoint": str(exp_dir / "best.pth"),
        "checkpoint_sha256": digest(exp_dir / "best.pth"),
        "calibration": str(calibration_path),
        "calibration_sha256": digest(calibration_path),
        "validation_state": str(val_path) if val_path else None,
        "test_state": str(test_path) if test_path else None,
        "validation_state_sha256": digest(val_path) if val_path else None,
        "test_state_sha256": digest(test_path) if test_path else None,
        "test_state_bytes": test_path.stat().st_size if test_path else 0,
        "selected_nodes": len(test_memory.node_ids) if test_memory else 0,
        "total_nodes": nodes,
        "validation_raw": metrics(val_prediction, val_truth),
        "test_raw": metrics(test_prediction, test_truth),
        "test_fixed_calibrated": metrics(calibrated, test_truth),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "dataset": CLI.dataset,
                      "seed": CLI.seed, "mode": CLI.mode,
                      "test_mae": report["test_raw"]["mae"],
                      "test_calibrated_mae": report["test_fixed_calibrated"]["mae"]}))


if __name__ == "__main__":
    main()
