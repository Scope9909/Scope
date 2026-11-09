"""Fixed Validation-only ValCal audit for one sensor-memory budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from calibrate_dualscale_smooth_tod import fit_and_apply  # noqa: E402
from calibrate_dualscale_quantile_residual import metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", required=True, type=Path)
    parser.add_argument("--full-memory", required=True, type=Path)
    parser.add_argument("--reference-calibration", required=True, type=Path)
    parser.add_argument("--selected-nodes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with np.load(args.no_memory, allow_pickle=False) as archive:
        no_memory = np.asarray(archive["val_prediction"], dtype=np.float64)
        truth = np.asarray(archive["val_truth"], dtype=np.float64)
        endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
    with np.load(args.full_memory, allow_pickle=False) as archive:
        full_memory = np.asarray(archive["val_prediction"], dtype=np.float64)
        if not np.array_equal(truth, archive["val_truth"]):
            raise ValueError("Validation targets differ")
        if not np.array_equal(endpoints, archive["val_endpoints"]):
            raise ValueError("Validation endpoints differ")
    with args.reference_calibration.open("r", encoding="utf-8") as stream:
        reference = json.load(stream)
    parameters = {
        key: reference[key]
        for key in (
            "num_bins", "smooth_shrinkage", "num_time_bins",
            "time_shrinkage", "slices_per_day",
        )
    }
    n = len(truth)
    folds = []
    for fold_index, (train_fraction, end_fraction) in enumerate(
        ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0))
    ):
        train_end = int(n * train_fraction)
        eval_end = int(n * end_fraction)
        reference_fold = reference["validation_folds"][fold_index]
        if (
            reference_fold["train_samples"] != train_end
            or reference_fold["evaluation_end"] != eval_end
        ):
            raise ValueError("Reference calibration fold boundaries differ")
        valid = np.round(truth[:train_end]) > 0
        denominator = valid.sum(axis=(0, 1))
        gain = (
            (np.abs(no_memory[:train_end] - truth[:train_end]) * valid).sum(axis=(0, 1))
            - (np.abs(full_memory[:train_end] - truth[:train_end]) * valid).sum(axis=(0, 1))
        ) / denominator
        selected = np.argsort(-gain, kind="stable")[:args.selected_nodes]
        combined = no_memory.copy()
        combined[:, :, selected] = full_memory[:, :, selected]
        smooth, calibrated, _ = fit_and_apply(
            combined[:train_end], truth[:train_end], endpoints[:train_end],
            combined[train_end:eval_end], endpoints[train_end:eval_end],
            **parameters,
        )
        target = truth[train_end:eval_end]
        raw = metrics(combined[train_end:eval_end], target)
        smooth_metrics = metrics(smooth, target)
        calibrated_metrics = metrics(calibrated, target)
        full_calibrated = reference_fold["smooth_plus_target_time"]
        fold = {
            "selection_samples": train_end,
            "evaluation_start": train_end,
            "evaluation_end": eval_end,
            "raw": raw,
            "smooth": smooth_metrics,
            "calibrated": calibrated_metrics,
            "full_memory_calibrated": full_calibrated,
            "calibrated_mae_minus_full_memory": (
                calibrated_metrics["mae"] - full_calibrated["mae"]
            ),
            "smooth_gate_passed": smooth_metrics["mae"] < raw["mae"],
            "target_time_gate_passed": calibrated_metrics["mae"] < smooth_metrics["mae"],
        }
        folds.append(fold)
        print(
            f"Fold {fold_index + 1}: sparse raw {raw['mae']:.4f}, "
            f"calibrated {calibrated_metrics['mae']:.4f}, "
            f"versus full calibrated {fold['calibrated_mae_minus_full_memory']:+.4f}",
            flush=True,
        )
    differences = [fold["calibrated_mae_minus_full_memory"] for fold in folds]
    report = {
        "protocol": "fixed_validation_only_calibration_of_sensor_sparse_memory",
        "selected_nodes": args.selected_nodes,
        "test_evaluated": False,
        "algebraic_sensor_mix_not_physical_sparse_forward": True,
        "reference_calibration": str(args.reference_calibration.resolve()),
        "folds": folds,
        "summary": {
            "mean_calibrated_mae_minus_full_memory": float(np.mean(differences)),
            "improved_fold_count": int(np.sum(np.asarray(differences) < 0)),
            "all_smooth_gates_passed": all(f["smooth_gate_passed"] for f in folds),
            "all_target_time_gates_passed": all(f["target_time_gate_passed"] for f in folds),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
