"""Fit and export the final Validation-only calibrator for frozen ALSC states.

This reruns Validation prediction with the existing seed-1 checkpoint and
selected state. It never loads Test predictions, states, or labels.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_lossless_state_codec as codec
import eval_hybrid_codec_test_once as previous
from calibrate_dualscale_smooth_tod import (
    fit_and_apply, smooth_correction_for, target_time_correction_for,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEST_REPORT = HERE / "packages/online/adaptive_lossless_test_20261023_v1/results.json"
VALIDATION = HERE / "packages/online/adaptive_lossless_val_20261023_v2"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_loaded(path: Path, prediction: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
    """Standalone inference: only the saved tables, predictions and endpoints."""
    with np.load(path, allow_pickle=False) as archive:
        centers = np.asarray(archive["centers"])
        residuals = np.asarray(archive["smooth_residuals"])
        time_table = np.asarray(archive["target_time_residuals"])
        smooth_shrinkage = float(archive["smooth_shrinkage"])
        time_shrinkage = float(archive["time_shrinkage"])
        slices_per_day = int(archive["slices_per_day"])
    smooth = prediction + smooth_shrinkage * smooth_correction_for(
        prediction, centers, residuals)
    return smooth + time_shrinkage * target_time_correction_for(
        endpoints, prediction.shape[1], time_table, slices_per_day)


def run_dataset(dataset: str, output: Path, expected: dict, manifest: dict) -> dict:
    runner = previous._load_runner(dataset)
    experiment, calibration_path, _, _, selector = runner.inputs()
    job = next(item for item in manifest["jobs"] if item["dataset"] == dataset)
    checkpoint = experiment / "best.pth"
    if checksum(checkpoint) != job["checkpoint_sha256"]:
        raise ValueError(f"{dataset}: checkpoint hash differs from frozen manifest")
    if checksum(calibration_path) != job["calibration_sha256"]:
        raise ValueError(f"{dataset}: calibration configuration hash differs")
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
        raise ValueError("Expected Student retrieval adapters")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("export_frozen_calibrators")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = runner.datasets_utils.load_dataset(args, logger)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(
        checkpoint, map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False
    with np.load(ROOT / "datasets" / dataset /
                 f"{dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz",
                 allow_pickle=False) as archive:
        endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
    state_path = VALIDATION / f"{dataset.lower()}_val_selected.alsc"
    memory = runner.OnlineResidualMemory(codec.load_runtime_state(state_path))
    if memory.cold_start_endpoint != int(endpoints[0]):
        raise ValueError("Validation state is misaligned")
    prediction, truth = runner.collect(
        loaders["val_loader"], endpoints, model, scaler, memory, device, 32)
    if len(prediction) != len(endpoints):
        raise AssertionError("Validation count includes padded duplicates")
    params = json.loads(calibration_path.read_text(encoding="utf-8"))
    keys = ("num_bins", "smooth_shrinkage", "num_time_bins",
            "time_shrinkage", "slices_per_day")
    kwargs = {key: params[key] for key in keys}
    _, fitted, tables = fit_and_apply(
        prediction, truth, endpoints, prediction, endpoints, **kwargs)
    centers, residuals, time_table = tables
    path = output / f"{dataset.lower()}_calibrator.npz"
    np.savez_compressed(
        path, centers=centers, smooth_residuals=residuals,
        target_time_residuals=time_table,
        **{key: np.asarray(value) for key, value in kwargs.items()},
    )
    restored = apply_loaded(path, prediction, endpoints)
    if not np.array_equal(restored, fitted, equal_nan=True):
        raise AssertionError(f"{dataset}: standalone calibrator prediction differs")
    raw = runner.metrics(prediction, truth)
    row = dict(dataset=dataset, seed=1, selector=selector,
               validation_count=len(prediction), validation_raw_metrics=raw,
               validation_calibrated_metrics=runner.metrics(restored, truth),
               checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checksum(checkpoint),
               calibration_config=str(calibration_path.resolve()),
               calibration_config_sha256=checksum(calibration_path),
               state=str(state_path.resolve()), state_sha256=checksum(state_path),
               calibrator=str(path.resolve()), calibrator_sha256=checksum(path),
               calibrator_bytes=path.stat().st_size,
               table_shapes={"centers": list(centers.shape),
                             "smooth_residuals": list(residuals.shape),
                             "target_time_residuals": list(time_table.shape)},
               standalone_reload_identical=True,
               previous_test_calibrated_mae=expected[dataset]["test_calibrated_metrics"]["mae"],
               test_not_rerun=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    report = json.loads(TEST_REPORT.read_text(encoding="utf-8"))
    if not report.get("complete"):
        raise ValueError("Previous Test report is incomplete")
    expected = {row["dataset"]: row for row in report["results"]}
    manifest = json.loads(previous.MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for dataset in previous.CASES:
        print(dataset, "Validation-only prediction and calibration", flush=True)
        row = run_dataset(dataset, cli.output, expected, manifest)
        rows.append(row)
        (cli.output / f"{dataset.lower()}_manifest.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8")
        print(dataset, "calibrator saved", row["calibrator_bytes"],
              "validation MAE", row["validation_calibrated_metrics"]["mae"], flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = dict(scope="Frozen selected-state Validation calibrator export",
                  validation_only=True, test_rerun=False,
                  all_reload_identical=all(r["standalone_reload_identical"] for r in rows),
                  total_calibrator_bytes=sum(r["calibrator_bytes"] for r in rows),
                  results=rows)
    (cli.output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("COMPLETE", result["total_calibrator_bytes"], flush=True)


if __name__ == "__main__":
    main()
