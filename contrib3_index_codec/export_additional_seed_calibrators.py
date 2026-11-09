"""Export Validation-fitted tables for reported PEMS04 s2/s3 and PEMS07 s2.

No Test data are opened. Existing Test results are only cited in the report.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from adaptive_lossless_state_codec import load_runtime_state
from calibrate_dualscale_smooth_tod import fit_and_apply
from export_frozen_calibrators import apply_loaded
from select_single_seed_validation import load_runner as load_pems07_runner

VAL = HERE / "packages/online/adaptive_lossless_val_v2"
CASES = (("PEMS04", 2), ("PEMS04", 3), ("PEMS07", 2))


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner(dataset, seed):
    if dataset == "PEMS07":
        return load_pems07_runner(seed)
    import importlib.util
    path = ROOT / "srm_test_once/run_frozen_srm_test_once.py"
    previous = sys.argv
    sys.argv = [str(path), "--dataset", dataset, "--seed", str(seed),
                "--mode", "sparse"]
    try:
        spec = importlib.util.spec_from_file_location(f"runner_{dataset}_{seed}", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        return runner
    finally:
        sys.argv = previous


def run_one(dataset, seed, output):
    runner = load_runner(dataset, seed)
    experiment, calibration_path, _, _, selector = runner.inputs()
    checkpoint = experiment / "best.pth"
    with (experiment / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(v) for v in args.k_hops.split(",")]
    args.pearson_adj = None
    if not (args.use_analog_lowrank_retrieval_residual and
            args.use_analog_exact_match_residual):
        raise ValueError("Expected frozen retrieval-aware Student")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("additional_calibrator_export")
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
    state_path = VAL / f"{dataset.lower()}_val_selected.alsc"
    memory = runner.OnlineResidualMemory(load_runtime_state(state_path))
    if memory.cold_start_endpoint != int(endpoints[0]):
        raise ValueError("Validation state misaligned")
    prediction, truth = runner.collect(
        loaders["val_loader"], endpoints, model, scaler, memory, device, 32)
    if len(prediction) != len(endpoints):
        raise AssertionError("Validation includes duplicated padded rows")
    params = json.loads(calibration_path.read_text(encoding="utf-8"))
    names = ("num_bins", "smooth_shrinkage", "num_time_bins",
             "time_shrinkage", "slices_per_day")
    kwargs = {name: params[name] for name in names}
    _, fitted, tables = fit_and_apply(
        prediction, truth, endpoints, prediction, endpoints, **kwargs)
    centers, residuals, time_table = tables
    cal_path = output / f"{dataset.lower()}_s{seed}_calibrator.npz"
    np.savez_compressed(cal_path, centers=centers, smooth_residuals=residuals,
                        target_time_residuals=time_table,
                        **{name: np.asarray(value) for name, value in kwargs.items()})
    reloaded = apply_loaded(cal_path, prediction, endpoints)
    if not np.array_equal(reloaded, fitted, equal_nan=True):
        raise AssertionError("Independent table reload changed Validation outputs")
    raw = runner.metrics(prediction, truth)
    if dataset == "PEMS04":
        previous = json.loads((ROOT / "srm_test_once" /
                               f"pems04_s{seed}_sparse_test_once.json").read_text())
        expected = previous["validation_raw"]["mae"]
    else:
        previous = json.loads((HERE / "packages/online/pems07_single_seed_selection_val.json").read_text())
        expected = next(row["validation_raw"]["mae"] for row in previous["rows"]
                        if row["seed"] == seed)
    if abs(raw["mae"] - expected) > 1e-8:
        raise AssertionError(f"{dataset} s{seed} Validation MAE not reproduced")
    return dict(dataset=dataset, seed=seed, selector=selector,
                checkpoint=str(checkpoint), checkpoint_sha256=hash_file(checkpoint),
                calibration_config=str(calibration_path),
                calibration_config_sha256=hash_file(calibration_path),
                validation_state=str(state_path), validation_state_sha256=hash_file(state_path),
                calibrator=str(cal_path), calibrator_sha256=hash_file(cal_path),
                calibrator_bytes=cal_path.stat().st_size,
                validation_count=len(truth), validation_raw=raw,
                validation_calibrated=runner.metrics(reloaded, truth),
                independent_reload_identical=True, test_data_used=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for dataset, seed in CASES:
        print("Exporting", dataset, seed, flush=True)
        row = run_one(dataset, seed, cli.output)
        rows.append(row)
        (cli.output / f"{dataset.lower()}_s{seed}_manifest.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved", row["calibrator_bytes"], "B", flush=True)
    (cli.output / "report.json").write_text(json.dumps(
        dict(scope="Additional reported seeds, Validation-only calibrator export",
             all_reload_identical=True, test_data_used=False, rows=rows),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
