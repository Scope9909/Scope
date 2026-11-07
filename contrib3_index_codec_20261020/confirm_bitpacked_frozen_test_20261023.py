"""One-time Test confirmation for the frozen exact delta-bitpacked state codec."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"
PROTOCOL = ONLINE / "bitpacked_codec_frozen_test_protocol_20261023.json"
OUT = ONLINE / "bitpacked_frozen_test_confirmation_20261023.json"
DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")


def load_module(name: str, path: Path, argv: list[str]):
    old = sys.argv
    try:
        sys.argv = argv
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old


eval_path = HERE / "eval_online_srm_codec.py"
base_path = HERE / "pilot_reversible_predictive_coding.py"
codec_path = HERE / "delta_bitpacked_state_codec.py"
runtime_path = HERE / "online_codec_runtime.py"
base_codec = load_module("frozen_test_base_codec", base_path, [str(base_path)])
codec = load_module("frozen_test_delta_codec", codec_path, [str(codec_path)])
runtime = load_module("frozen_test_online_runtime", runtime_path, [str(runtime_path)])


def run_dataset(dataset: str) -> dict:
    eval_module = load_module(
        "frozen_test_eval", eval_path,
        [str(eval_path), dataset, "--seed", "1", "--validation-only"])
    runner = eval_module.runner
    exp_dir, calibration_path, val_path, test_path, _ = runner.inputs()
    if test_path is None:
        raise ValueError(f"Missing fixed Test state for {dataset}")
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
        raise ValueError("Expected frozen Student low-rank plus exact-history checkpoint")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger(f"frozen_bitpack_test_{dataset}")
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

    index_path = (HERE.parent / "datasets" / dataset /
                  f"{dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz")
    with np.load(index_path, allow_pickle=False) as archive:
        val_endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
        test_endpoints = np.asarray(archive["idx_test"], dtype=np.int64)

    val_package = (ONLINE / "bitpacked_val_20261023" /
                   f"{dataset.lower()}_s1_val_delta_bitpacked.npz")
    packed_val_state = runtime.decode_online_state(val_package)
    with np.load(val_path, allow_pickle=False) as archive:
        baseline_val_state = {key: archive[key] for key in archive.files}
    baseline_val_memory = runner.OnlineResidualMemory(baseline_val_state)
    packed_val_memory = runner.OnlineResidualMemory(packed_val_state)

    # The Test state is encoded with the already frozen Validation-selected codec.
    with tempfile.TemporaryDirectory(prefix=f"{dataset}_frozen_test_", dir=ONLINE) as td:
        packed_test_path = Path(td) / "test_state.npz"
        codec.save(test_path, packed_test_path)
        packed_test_state = runtime.decode_online_state(packed_test_path)
        with np.load(test_path, allow_pickle=False) as archive:
            baseline_test_state = {key: archive[key] for key in archive.files}
        baseline_test_memory = runner.OnlineResidualMemory(baseline_test_state)
        packed_test_memory = runner.OnlineResidualMemory(packed_test_state)
        if (baseline_test_memory.cold_start_endpoint is not None or
                packed_test_memory.cold_start_endpoint is not None):
            raise ValueError("Test state must be fully observed Train state")

        start = time.perf_counter()
        base_val_prediction, val_truth = runner.collect(
            loaders["val_loader"], val_endpoints, model, scaler,
            baseline_val_memory, device, 32)
        baseline_val_seconds = time.perf_counter() - start
        start = time.perf_counter()
        packed_val_prediction, packed_val_truth = runner.collect(
            loaders["val_loader"], val_endpoints, model, scaler,
            packed_val_memory, device, 32)
        packed_val_seconds = time.perf_counter() - start
        if not np.array_equal(val_truth, packed_val_truth):
            raise AssertionError("Validation truth mismatch")

        start = time.perf_counter()
        base_test_prediction, test_truth = runner.collect(
            loaders["test_loader"], test_endpoints, model, scaler,
            baseline_test_memory, device, 32)
        baseline_test_seconds = time.perf_counter() - start
        start = time.perf_counter()
        packed_test_prediction, packed_test_truth = runner.collect(
            loaders["test_loader"], test_endpoints, model, scaler,
            packed_test_memory, device, 32)
        packed_test_seconds = time.perf_counter() - start
        if not np.array_equal(test_truth, packed_test_truth):
            raise AssertionError("Test truth mismatch")

        parameters = json.loads(calibration_path.read_text(encoding="utf-8"))
        kwargs = {key: parameters[key] for key in (
            "num_bins", "smooth_shrinkage", "num_time_bins",
            "time_shrinkage", "slices_per_day")}
        _, base_calibrated, _ = runner.fit_and_apply(
            base_val_prediction, val_truth, val_endpoints,
            base_test_prediction, test_endpoints, **kwargs)
        _, packed_calibrated, _ = runner.fit_and_apply(
            packed_val_prediction, packed_val_truth, val_endpoints,
            packed_test_prediction, test_endpoints, **kwargs)

        base_lossless, _ = base_codec.normalize_state(test_path)
        base_lossless_blob, _ = base_codec.best_bytes(base_lossless)
        return {
            "dataset": dataset,
            "seed": 1,
            "test_samples": int(len(test_endpoints)),
            "baseline_test_raw_mae": runner.metrics(base_test_prediction, test_truth)["mae"],
            "bitpacked_test_raw_mae": runner.metrics(packed_test_prediction, packed_test_truth)["mae"],
            "baseline_test_calibrated_mae": runner.metrics(base_calibrated, test_truth)["mae"],
            "bitpacked_test_calibrated_mae": runner.metrics(packed_calibrated, test_truth)["mae"],
            "raw_prediction_arrays_bitwise_equal": np.array_equal(
                base_test_prediction, packed_test_prediction),
            "val_prediction_arrays_bitwise_equal": np.array_equal(
                base_val_prediction, packed_val_prediction),
            "test_state_strong_lossless_bytes": len(base_lossless_blob),
            "test_state_bitpacked_bytes": packed_test_path.stat().st_size,
            "test_state_saved_bytes": len(base_lossless_blob) - packed_test_path.stat().st_size,
            "baseline_full_test_seconds": baseline_test_seconds,
            "bitpacked_full_test_seconds": packed_test_seconds,
            "baseline_full_val_seconds": baseline_val_seconds,
            "bitpacked_full_val_seconds": packed_val_seconds,
            "test_evaluated_once": True,
        }


def preflight() -> dict:
    results = []
    for dataset in DATASETS:
        eval_module = load_module(
            "frozen_test_preflight", eval_path,
            [str(eval_path), dataset, "--seed", "1", "--validation-only"])
        _, _, _, test_path, _ = eval_module.runner.inputs()
        if test_path is None:
            raise ValueError(f"Missing Test state for {dataset}")
        with tempfile.TemporaryDirectory(prefix=f"{dataset}_test_preflight_", dir=ONLINE) as td:
            packed = Path(td) / "state.npz"
            size = codec.save(test_path, packed)["bytes"]
            decoded = runtime.decode_online_state(packed)
            with np.load(test_path, allow_pickle=False) as archive:
                original = {key: archive[key] for key in archive.files}
            if decoded.keys() != original.keys():
                raise AssertionError(f"{dataset}: Test-state fields changed")
            for key, value in original.items():
                expected = np.asarray(value, dtype=np.float32) if key == "train_raw" else value
                if not np.array_equal(decoded[key], expected):
                    raise AssertionError(f"{dataset}: Test-state mismatch at {key}")
            baseline_state, _ = base_codec.normalize_state(test_path)
            baseline_blob, _ = base_codec.best_bytes(baseline_state)
            results.append({"dataset": dataset, "test_state_path": str(test_path),
                            "strong_lossless_bytes": len(baseline_blob),
                            "candidate_bytes": size,
                            "exact_state_roundtrip": True,
                            "test_labels_read": False})
    return {"preflight_only": True, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    options = parser.parse_args()
    if options.preflight:
        print(json.dumps(preflight(), indent=2))
        return
    if OUT.exists():
        raise FileExistsError(f"Refusing to repeat frozen Test confirmation: {OUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["method"]["codec_id"] != codec.CODEC_ID:
        raise ValueError("Frozen protocol and implementation codec id differ")
    results = [run_dataset(dataset) for dataset in DATASETS]
    result = {
        "title": "Frozen exact delta-bitpacked codec Test confirmation",
        "date": "2026-10-23",
        "protocol": str(PROTOCOL),
        "test_selection_or_tuning": False,
        "results": results,
        "limitations": ["Seed 1 only", "One Test pass per state; no post-Test tuning", "Single-run timings are diagnostic"],
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
