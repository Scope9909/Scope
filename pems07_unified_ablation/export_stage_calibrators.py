"""Fit each PEMS07 stage's deployable calibration table on Validation only."""
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


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CODEC_DIR = ROOT / "contrib3_index_codec"
sys.path.insert(0, str(CODEC_DIR))
import adaptive_lossless_state_codec as codec  # noqa: E402
import eval_hybrid_codec_test_once as hybrid  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(stage: str):
    runner = hybrid._load_runner("PEMS07")
    student_dir, calibration_path, _, _, _ = runner.inputs()
    experiment = (ROOT / "EXP_experiments" / "TLAST_improved" / "PEMS07" /
                  "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint"
                  if stage == "teacher_base" else student_dir)
    with (experiment / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    args.pearson_adj = None
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    flags = {key: value for key, value in vars(args).items()
             if key.startswith("use_analog_")}
    for key in flags:
        setattr(args, key, 0)
    logger = logging.getLogger("pems07_stage_calibrator")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = runner.datasets_utils.load_dataset(args, logger)
    for key, value in flags.items():
        setattr(args, key, value)
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    checkpoint = experiment / "best.pth"
    engine.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    for key in flags:
        if hasattr(model, key):
            setattr(model, key, False)
    return runner, model, loaders, scaler, device, calibration_path, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("teacher_base", "student_base", "full_exact", "srm_exact"))
    cli = parser.parse_args()
    output = HERE / f"pems07_{cli.stage}_calibrator.npz"
    report_path = HERE / f"pems07_{cli.stage}_calibrator.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("Stage calibrator exists")
    runner, model, loaders, scaler, device, config, checkpoint = setup(cli.stage)
    with np.load(ROOT / "datasets" / "PEMS07" / "PEMS07_12to12_index_scaler.npz",
                 allow_pickle=False) as archive:
        endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
    state_paths = {
        "full_exact": HERE / "pems07_full_exact_val.alsc",
        "srm_exact": (CODEC_DIR / "packages" / "online" /
                      "adaptive_lossless_val_v2" / "pems07_val_exact.alsc"),
    }
    state_path = state_paths.get(cli.stage)
    memory = (runner.OnlineResidualMemory(codec.load_runtime_state(state_path))
              if state_path is not None else None)
    print(f"{cli.stage}: Validation prediction", flush=True)
    prediction, truth = runner.collect(
        loaders["val_loader"], endpoints, model, scaler, memory, device, 32)
    params = json.loads(config.read_text(encoding="utf-8"))
    kwargs = {key: params[key] for key in (
        "num_bins", "smooth_shrinkage", "num_time_bins", "time_shrinkage", "slices_per_day")}
    _, fitted, tables = runner.fit_and_apply(
        prediction, truth, endpoints, prediction, endpoints, **kwargs)
    centers, residuals, time_table = tables
    np.savez_compressed(
        output, centers=centers, smooth_residuals=residuals,
        target_time_residuals=time_table,
        **{key: np.asarray(value) for key, value in kwargs.items()})
    with np.load(output, allow_pickle=False) as archive:
        assert np.array_equal(archive["centers"], centers)
        assert np.array_equal(archive["smooth_residuals"], residuals)
        assert np.array_equal(archive["target_time_residuals"], time_table)
    report = {
        "stage": cli.stage, "dataset": "PEMS07", "seed": 1,
        "fitted_on": "Validation", "validation_count": len(prediction),
        "validation_raw_metrics": runner.metrics(prediction, truth),
        "validation_fitted_metrics": runner.metrics(fitted, truth),
        "checkpoint_sha256": sha256(checkpoint),
        "state_sha256": sha256(state_path) if state_path else None,
        "calibrator_sha256": sha256(output), "calibrator_bytes": output.stat().st_size,
        "table_shapes": [list(item.shape) for item in tables],
        "test_not_evaluated": True,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
