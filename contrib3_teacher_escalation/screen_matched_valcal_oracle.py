"""Validation-only headroom screen for matched-calibrated Student and Teacher.

Each model fits the same frozen ValCal recipe on the first half of Validation.
The final quarter, separated by a 12-endpoint embargo, is used only to measure
the maximum possible gain from routing at most 5/10/20 percent of windows.
No Test prediction or label key is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "matched_valcal_oracle_screen.json"
sys.path.insert(0, str(ROOT))
from calibrate_dualscale_smooth_tod import fit_and_apply


SOURCES = {
    "PEMS03": {
        "student": "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz",
        "calibration": "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/calibration.json",
        "teacher": "datasets/PEMS03/PEMS03_12to12_parent_predictions_joint3.npz",
    },
    "PEMS04": {
        "student": "paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_s1/selected_predictions.npz",
        "calibration": "paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_s1_valcal/calibration.json",
        "teacher": "datasets/PEMS04/PEMS04_12to12_parent_predictions_joint3_strict.npz",
    },
    "PEMS07": {
        "student": "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz",
        "calibration": "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/calibration.json",
        "teacher": "datasets/PEMS07/PEMS07_12to12_parent_predictions_joint3.npz",
    },
    "PEMS08": {
        "student": "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz",
        "calibration": "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/calibration.json",
        "teacher": "datasets/PEMS08/PEMS08_12to12_parent_predictions_joint3.npz",
    },
}
CALIBRATION_KEYS = (
    "num_bins",
    "smooth_shrinkage",
    "num_time_bins",
    "time_shrinkage",
    "slices_per_day",
)


def official_error(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(truth) & (np.rint(truth) > 0)
    return (
        np.where(mask, np.abs(prediction - truth), 0.0).sum(axis=2, dtype=np.float64),
        mask.sum(axis=2, dtype=np.float64),
    )


def main() -> None:
    report = {
        "protocol": "matched_ValCal_early_Validation_fit_late_Validation_oracle_v1",
        "test_keys_read": False,
        "metric": "mean of the 12 masked horizon MAEs",
        "caveat": (
            "The frozen ValCal hyperparameters were previously chosen with Validation; "
            "this is a feasibility screen, not independent final confirmation."
        ),
        "datasets": {},
    }
    for dataset, paths in SOURCES.items():
        with np.load(ROOT / paths["student"], allow_pickle=False) as archive:
            endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
            student = np.asarray(archive["val_prediction"], dtype=np.float32)
            truth = np.asarray(archive["val_truth"], dtype=np.float32)
        with np.load(ROOT / paths["teacher"], allow_pickle=False) as archive:
            teacher_z = np.asarray(archive["teacher_pred_val"], dtype=np.float32)[..., 0]
            teacher_mean = float(np.asarray(archive["scaler_mean"]))
            teacher_std = float(np.asarray(archive["scaler_std"]))
        if len(teacher_z) < len(endpoints):
            raise ValueError(f"{dataset}: Teacher Validation cache is too short")
        teacher = teacher_z[: len(endpoints)] * teacher_std + teacher_mean
        del teacher_z
        if not (student.shape == teacher.shape == truth.shape):
            raise ValueError(f"{dataset}: prediction and truth shape mismatch")
        with np.load(ROOT / f"datasets/{dataset}/{dataset}_12to12_index_scaler.npz") as archive:
            official_endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
        if not np.array_equal(endpoints, official_endpoints):
            raise ValueError(f"{dataset}: Validation endpoint mismatch")
        if not np.all(np.diff(endpoints) == 1):
            raise ValueError(f"{dataset}: endpoints are not consecutive")

        n = len(endpoints)
        fit = slice(0, n // 2)
        evaluation = slice(3 * n // 4 + 12, n)
        if endpoints[evaluation.start] + 1 <= endpoints[fit.stop - 1] + 12:
            raise ValueError(f"{dataset}: future labels overlap across fit/evaluation")
        config = json.loads((ROOT / paths["calibration"]).read_text(encoding="utf-8"))
        parameters = {key: config[key] for key in CALIBRATION_KEYS}
        print(f"{dataset}: fit matched Student ValCal", flush=True)
        _, student_calibrated, _ = fit_and_apply(
            student[fit].astype(np.float64),
            truth[fit].astype(np.float64),
            endpoints[fit],
            student[evaluation].astype(np.float64),
            endpoints[evaluation],
            **parameters,
        )
        print(f"{dataset}: fit matched Teacher ValCal", flush=True)
        _, teacher_calibrated, _ = fit_and_apply(
            teacher[fit].astype(np.float64),
            truth[fit].astype(np.float64),
            endpoints[fit],
            teacher[evaluation].astype(np.float64),
            endpoints[evaluation],
            **parameters,
        )
        evaluation_truth = truth[evaluation]
        student_error, counts = official_error(student_calibrated, evaluation_truth)
        teacher_error, teacher_counts = official_error(teacher_calibrated, evaluation_truth)
        if not np.array_equal(counts, teacher_counts):
            raise AssertionError(f"{dataset}: masks differ")
        horizon_counts = counts.sum(axis=0)
        student_mae = float(np.mean(student_error.sum(axis=0) / horizon_counts))
        teacher_mae = float(np.mean(teacher_error.sum(axis=0) / horizon_counts))
        utility = np.mean((student_error - teacher_error) / horizon_counts[None, :], axis=1)
        budgets = {}
        for budget in (0.05, 0.10, 0.20):
            maximum = int(np.floor(budget * len(utility)))
            top = np.sort(utility)[-maximum:]
            positive = top[top > 0.0]
            budgets[str(budget)] = {
                "max_teacher_windows": maximum,
                "oracle_chosen_windows": int(len(positive)),
                "oracle_gain_at_most_budget": float(positive.sum()),
            }
        report["datasets"][dataset] = {
            "sources": paths,
            "calibration_params": parameters,
            "fit_indices": [fit.start, fit.stop - 1],
            "evaluation_indices": [evaluation.start, n - 1],
            "evaluation_windows": int(len(utility)),
            "student_matched_valcal_mae": student_mae,
            "teacher_matched_valcal_mae": teacher_mae,
            "teacher_always_gain": student_mae - teacher_mae,
            "teacher_better_window_fraction": float(np.mean(utility > 0)),
            "budgets": budgets,
        }
        print(
            f"{dataset}: Student={student_mae:.6f}, Teacher={teacher_mae:.6f}, "
            f"Oracle<=10%={budgets['0.1']['oracle_gain_at_most_budget']:.6f}",
            flush=True,
        )
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
