"""Evaluate one frozen sparse-node plan with fixed Validation-only calibration.

Selection must already be frozen in plan.json. This script never reads Test, does
not refit any hyperparameters, and materializes no large prediction archive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_dualscale_smooth_tod import fit_and_apply  # noqa: E402
from calibrate_dualscale_quantile_residual import metrics  # noqa: E402
from screen_heterogeneous_memory import FOLDS, load  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--backbone-predictions", type=Path, required=True)
    parser.add_argument("--int8-predictions", type=Path, required=True)
    parser.add_argument("--reference-calibration", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    choices = np.asarray(plan["folds"][0]["heterogeneous_mode_by_node"])
    if not np.all((choices == 0) | (choices == 1)):
        raise ValueError("This evaluator accepts OFF/INT8 plans only")
    backbone, truth, endpoints = load(args.backbone_predictions)
    full, truth2, endpoints2 = load(args.int8_predictions)
    if not (backbone.shape == full.shape == truth.shape and
            np.array_equal(truth, truth2) and np.array_equal(endpoints, endpoints2)
            and backbone.shape[2] == len(choices)):
        raise ValueError("Predictions/truth/endpoints/plan alignment failed")
    native = json.loads(args.native_report.read_text(encoding="utf-8"))
    expected_validation_archive_bytes = plan.get("physical_archive_bytes_in_memory")
    if (not native.get("passed_mean_prediction_and_mae_parity") or
            native["selected_nodes"] != int((choices == 1).sum()) or
            (expected_validation_archive_bytes is not None and
             native["cache_bytes"] != expected_validation_archive_bytes)):
        raise ValueError("Native parity or archive-byte check failed")
    reference = json.loads(args.reference_calibration.read_text(encoding="utf-8"))
    parameters = {key: reference[key] for key in
                  ("num_bins", "smooth_shrinkage", "num_time_bins",
                   "time_shrinkage", "slices_per_day")}
    selected = np.where(choices[None, None, :] == 1, full, backbone).astype(np.float64)
    truth = truth.astype(np.float64)
    rows = []
    for i, (start_fraction, end_fraction) in enumerate(FOLDS):
        fit_end = int(len(truth) * start_fraction)
        end = int(len(truth) * end_fraction)
        reference_fold = reference["validation_folds"][i]
        if (fit_end != reference_fold["train_samples"] or
                end != reference_fold["evaluation_end"]):
            raise ValueError("Reference calibration fold boundaries differ")
        _, calibrated, _ = fit_and_apply(
            selected[:fit_end], truth[:fit_end], endpoints[:fit_end],
            selected[fit_end:end], endpoints[fit_end:end], **parameters)
        rows.append({"fit_end": fit_end, "eval_end": end,
                     "raw_mae": metrics(selected[fit_end:end], truth[fit_end:end])["mae"],
                     "fixed_calibrated_mae": metrics(calibrated, truth[fit_end:end])["mae"]})
    report = {
        "protocol": "frozen_plan_expanding_time_validation_fixed_calibration",
        "test_evaluated": False,
        "selected_nodes": int((choices == 1).sum()),
        "physical_archive_bytes": native["cache_bytes"],
        "online_state_bytes": plan.get("online_state_bytes"),
        "selected_total_bytes": plan.get("selected_total_bytes"),
        "native_algebraic_parity": True,
        "reference_calibration": str(args.reference_calibration.resolve()),
        "folds": rows,
        "mean_raw_mae": float(np.mean([row["raw_mae"] for row in rows])),
        "mean_fixed_calibrated_mae": float(np.mean(
            [row["fixed_calibrated_mae"] for row in rows])),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
