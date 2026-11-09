"""One-time final evaluation of the Validation-frozen hybrid codec.

Selection is fixed before Test: PEMS03/07 use mixed Q8 + exact delta packing;
PEMS04/08 use exact delta packing only. This script refuses to overwrite its
freeze manifest or result, and never compares/selects methods on Test.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ONLINE = HERE / "packages" / "online"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from composite_q8_delta_codec import decode_composite_online_state, pack_state, _serialize
from delta_bitpacked_state_codec import encode_state, serialize
from online_codec_runtime import decode_online_state


CASES = {
    "PEMS03": {
        "choice": "Q8 + exact delta-bitpacking",
        "val_source": "pems03_s1_val_fixed_22_idx.npz",
        "test_source": "pems03_s1_test_fixed_22_idx.npz",
        "val_package": "pems03_s1_val_q8_delta_pilot.npz",
        "test_package": "pems03_s1_test_hybrid_q8_delta_once.npz",
        "roundtrip_test_source": "pems03_s1_test_fixed_22_idx.npz",
        "kind": "mixed",
    },
    "PEMS04": {
        "choice": "strong-lossless + exact delta-bitpacking",
        "val_source": "pems04_s1_val_u16.npz",
        "test_source": "pems04_s1_test_u16.npz",
        "val_package": "bitpacked_val/pems04_s1_val_delta_bitpacked.npz",
        "test_package": "pems04_s1_test_delta_bitpacked_once.npz",
        "roundtrip_test_source": "pems04_s1_test_u16.npz",
        "kind": "exact",
    },
    "PEMS07": {
        "choice": "Q8 + exact delta-bitpacking",
        "val_source": "pems07_s1_val_fixed_117_idx.npz",
        "test_source": "pems07_s1_test_fixed_117_idx.npz",
        "val_package": "pems07_s1_val_q8_delta_pilot.npz",
        "test_package": "pems07_s1_test_hybrid_q8_delta_once.npz",
        "roundtrip_test_source": "pems07_s1_test_fixed_117_idx.npz",
        "kind": "mixed",
    },
    "PEMS08": {
        "choice": "strong-lossless + exact delta-bitpacking",
        "val_source": "pems08_s1_val_u16.npz",
        "test_source": "pems08_s1_test_u16.npz",
        "val_package": "bitpacked_val/pems08_s1_val_delta_bitpacked.npz",
        "test_package": "pems08_s1_test_delta_bitpacked_once.npz",
        "roundtrip_test_source": "pems08_s1_test_u16.npz",
        "kind": "exact",
    },
}
MANIFEST = ONLINE / "hybrid_codec_test_once_frozen_manifest_final.json"
REPORT = ONLINE / "hybrid_codec_test_once_results_final.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runner(dataset: str):
    runner_path = ROOT / "srm_test_once" / "run_frozen_srm_test_once.py"
    sys.argv = [str(runner_path), "--dataset", dataset, "--seed", "1", "--mode", "sparse"]
    spec = importlib.util.spec_from_file_location("hybrid_frozen_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sys.argv = [__file__]
    return runner


def _make_manifest() -> dict:
    jobs = []
    for dataset, case in CASES.items():
        runner = _load_runner(dataset)
        exp_dir, calibration, val_path, test_path, _ = runner.inputs()
        jobs.append({
            "dataset": dataset,
            "seed": 1,
            "frozen_choice": case["choice"],
            "validation_package": str(ONLINE / case["val_package"]),
            "validation_package_sha256": sha256(ONLINE / case["val_package"]),
            "test_state_source": str(ONLINE / case["test_source"]),
            "test_package_output": str(ONLINE / case["test_package"]),
            "checkpoint": str(exp_dir / "best.pth"),
            "checkpoint_sha256": sha256(exp_dir / "best.pth"),
            "calibration": str(calibration),
            "calibration_sha256": sha256(calibration),
            "validation_state": str(val_path),
            "test_state_reference": str(test_path),
        })
    return {
        "protocol": "hybrid_q8_delta_validation_frozen_test_once_v1",
        "frozen_before_test": True,
        "selection_data": "Validation package bytes and frozen Validation error gate only",
        "test_role": "one-time final confirmation; no post-Test selection or tuning",
        "test_passes_per_dataset": 1,
        "batch_size": 32,
        "method_source_sha256": sha256(Path(__file__)),
        "codec_source_sha256": sha256(HERE / "composite_q8_delta_codec.py"),
        "delta_codec_source_sha256": sha256(HERE / "delta_bitpacked_state_codec.py"),
        "jobs": jobs,
    }


def _write_selected_test_package(dataset: str, case: dict) -> Path:
    source = ONLINE / case["test_source"]
    target = ONLINE / case["test_package"]
    if not target.exists():
        with np.load(source, allow_pickle=False) as archive:
            state = {key: archive[key] for key in archive.files}
        if case["kind"] == "mixed":
            packed = pack_state(state)
            blob, _ = _serialize(packed)
        else:
            packed = encode_state(state)
            blob, _ = serialize(packed)
        target.write_bytes(blob)
    expected = decode_online_state(ONLINE / case["roundtrip_test_source"])
    actual = (decode_composite_online_state(target) if case["kind"] == "mixed"
              else decode_online_state(target))
    mismatches = [key for key in expected
                  if key not in actual or not np.array_equal(expected[key], actual[key])]
    if mismatches:
        target.unlink(missing_ok=True)
        raise AssertionError(f"Test package decode mismatch for {dataset}: {mismatches}")
    return target


def _eval_one(dataset: str, case: dict) -> dict:
    runner = _load_runner(dataset)
    exp_dir, calibration_path, val_path, _, selector = runner.inputs()
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
    logger = logging.getLogger("hybrid_codec_test_once")
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

    index_path = (ROOT / "datasets" / dataset /
                  f"{dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz")
    with np.load(index_path, allow_pickle=False) as index:
        val_endpoints = np.asarray(index["idx_val"], dtype=np.int64)
        test_endpoints = np.asarray(index["idx_test"], dtype=np.int64)

    val_file = ONLINE / case["val_package"]
    test_file = ONLINE / case["test_package"]
    val_state = (decode_composite_online_state(val_file) if case["kind"] == "mixed"
                 else decode_online_state(val_file))
    test_state = (decode_composite_online_state(test_file) if case["kind"] == "mixed"
                  else decode_online_state(test_file))
    val_memory = runner.OnlineResidualMemory(val_state)
    test_memory = runner.OnlineResidualMemory(test_state)
    if val_memory.cold_start_endpoint != int(val_endpoints[0]):
        raise ValueError(f"Validation alignment mismatch for {dataset}")
    if test_memory.cold_start_endpoint is not None:
        raise ValueError(f"Test state must be fully observed for {dataset}")

    # Measure Validation baseline and selected final method; only selected final
    # method is evaluated on Test, exactly once.
    raw_val_memory = runner.OnlineResidualMemory.from_npz(val_path)
    base_val_pred, val_truth = runner.collect(
        loaders["val_loader"], val_endpoints, model, scaler, raw_val_memory, device, 32)
    selected_val_pred, selected_val_truth = runner.collect(
        loaders["val_loader"], val_endpoints, model, scaler, val_memory, device, 32)
    if not np.array_equal(val_truth, selected_val_truth):
        raise AssertionError("Validation truth alignment changed")
    base_val_mae = runner.metrics(base_val_pred, val_truth)["mae"]
    selected_val_mae = runner.metrics(selected_val_pred, val_truth)["mae"]
    test_pred, test_truth = runner.collect(
        loaders["test_loader"], test_endpoints, model, scaler, test_memory, device, 32)
    parameters = json.loads(calibration_path.read_text(encoding="utf-8"))
    kwargs = {key: parameters[key] for key in (
        "num_bins", "smooth_shrinkage", "num_time_bins",
        "time_shrinkage", "slices_per_day")}
    _, calibrated, _ = runner.fit_and_apply(
        selected_val_pred, val_truth, val_endpoints,
        test_pred, test_endpoints, **kwargs)
    return {
        "dataset": dataset,
        "seed": 1,
        "selector": selector,
        "frozen_choice": case["choice"],
        "validation_package_bytes": val_file.stat().st_size,
        "test_package_bytes": test_file.stat().st_size,
        "validation_baseline_mae": base_val_mae,
        "validation_final_mae": selected_val_mae,
        "validation_delta_mae": selected_val_mae - base_val_mae,
        "test_raw_mae": runner.metrics(test_pred, test_truth)["mae"],
        "test_val_calibrated_mae": runner.metrics(calibrated, test_truth)["mae"],
        "validation_count": len(val_truth),
        "test_count": len(test_truth),
        "test_evaluated_once": True,
        "test_state_roundtrip_bitwise_equal": True,
    }


def main() -> None:
    if MANIFEST.exists() or REPORT.exists():
        raise FileExistsError("Final one-time hybrid Test manifest/result already exists; refusing rerun")
    manifest = _make_manifest()
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    for dataset, case in CASES.items():
        result_path = ONLINE / f"hybrid_codec_test_once_{dataset.lower()}_20261023_final.json"
        started_path = ONLINE / f"hybrid_codec_test_once_{dataset.lower()}_20261023_started.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        if started_path.exists():
            raise RuntimeError(f"{dataset} Test pass was interrupted after starting; refusing to repeat it")
        _write_selected_test_package(dataset, case)
        started_path.write_text(json.dumps({"dataset": dataset, "test_pass_started": True}), encoding="utf-8")
        print(f"Frozen Test evaluation starting: {dataset}", flush=True)
        result = _eval_one(dataset, case)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        started_path.unlink()
        results.append(result)
    report = {
        "scope": "One-time Test confirmation of Validation-frozen final hybrid codec",
        "selection_or_tuning_on_test": False,
        "test_repeat_allowed": False,
        "frozen_manifest": str(MANIFEST),
        "results": results,
        "total_validation_package_bytes": sum(x["validation_package_bytes"] for x in results),
        "total_test_package_bytes": sum(x["test_package_bytes"] for x in results),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
