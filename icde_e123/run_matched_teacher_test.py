"""Frozen E1 calibration protocol applied fairly to PEMS04/07 Test.

The calibration hyperparameters are taken from the existing Student ValCal
artifact.  Teacher and Student each fit tables on their own full Validation
predictions and use the same Test truth/endpoints.  Existing Student ValCal
predictions must reproduce exactly after float32 serialization.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "matched_teacher_test_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_dualscale_quantile_residual import metrics
from calibrate_dualscale_smooth_tod import fit_and_apply
from run_e1_e3 import starts, sums, teacher

SOURCES = {
    "PEMS04": (
        "paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_s1_valcal",
        "datasets/PEMS04/PEMS04_12to12_parent_predictions_joint3_strict.npz",
    ),
    "PEMS07": (
        "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal",
        "datasets/PEMS07/PEMS07_12to12_parent_predictions_joint3.npz",
    ),
}


def paired_ci(teacher_pred, student_pred, truth, endpoints, dataset):
    length, repeats = 288, 5000
    possible = starts(endpoints, length)
    if not len(possible):
        raise ValueError(f"{dataset}: no continuous {length}-endpoint blocks")
    parent_sum, counts = sums(teacher_pred, truth)
    student_sum, student_counts = sums(student_pred, truth)
    if not np.array_equal(counts, student_counts):
        raise ValueError("Paired Test masks differ")
    n = len(truth)
    observed = float(np.mean(parent_sum.sum(axis=0) / counts.sum(axis=0)
                             - student_sum.sum(axis=0) / counts.sum(axis=0)))
    rng = np.random.default_rng(20261016 + (0 if dataset == "PEMS04" else 1))
    draws = np.empty(repeats)
    offsets = np.arange(length)
    number = (n + length - 1) // length
    for begin in range(0, repeats, 20):
        batch = min(20, repeats - begin)
        chosen = rng.choice(possible, size=(batch, number), replace=True)
        ix = (chosen[:, :, None] + offsets).reshape(batch, -1)[:, :n]
        weights = counts[ix].sum(axis=1)
        draws[begin:begin + batch] = np.mean(
            parent_sum[ix].sum(axis=1) / weights
            - student_sum[ix].sum(axis=1) / weights, axis=1)
    return {
        "difference": "Teacher+matched-ValCal minus Student+matched-ValCal",
        "observed_mae_difference": observed,
        "block_length": length,
        "repeats": repeats,
        "valid_block_starts": len(possible),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "probability_difference_le_zero": float(np.mean(draws <= 0)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": "E1_frozen_equal_ValCal_fair_Test_v1",
        "calibration_choice": "Existing Student Validation-selected parameters; no new Test tuning",
        "metric": "official round(truth)>0 masked per-horizon MAE then horizon mean",
        "datasets": {},
    }
    for dataset, (cal_rel, parent_rel) in SOURCES.items():
        cal_dir = ROOT / cal_rel
        cal = json.loads((cal_dir / "calibration.json").read_text(encoding="utf-8"))
        params = {key: cal[key] for key in ("num_bins", "smooth_shrinkage",
                  "num_time_bins", "time_shrinkage", "slices_per_day")}
        student_raw_path = Path(cal["source_predictions"])
        if not student_raw_path.exists():
            student_raw_path = ROOT / student_raw_path
        with np.load(student_raw_path, allow_pickle=False) as a:
            student_val = np.asarray(a["val_prediction"], dtype=np.float64)
            val_truth = np.asarray(a["val_truth"], dtype=np.float64)
            val_endpoints = np.asarray(a["val_endpoints"], dtype=np.int64)
            student_test = np.asarray(a["test_prediction"], dtype=np.float64)
            test_truth = np.asarray(a["test_truth"], dtype=np.float64)
            test_endpoints = np.asarray(a["test_endpoints"], dtype=np.int64)
        teacher_source = ROOT / parent_rel
        teacher_val, val_padding = teacher(teacher_source, "val", len(val_truth))
        teacher_test, test_padding = teacher(teacher_source, "test", len(test_truth))
        if not (student_val.shape == teacher_val.shape == val_truth.shape
                and student_test.shape == teacher_test.shape == test_truth.shape):
            raise ValueError(f"{dataset}: prediction/truth shapes are not aligned")
        if not (np.all(np.diff(val_endpoints) > 0)
                and np.all(np.diff(test_endpoints) > 0)):
            raise ValueError(f"{dataset}: non-monotonic endpoints")
        print(f"{dataset} Teacher fit/apply", flush=True)
        _, teacher_cal, teacher_tables = fit_and_apply(
            teacher_val.astype(np.float64), val_truth, val_endpoints,
            teacher_test.astype(np.float64), test_endpoints, **params)
        print(f"{dataset} Student fit/apply", flush=True)
        _, student_cal, student_tables = fit_and_apply(
            student_val, val_truth, val_endpoints,
            student_test, test_endpoints, **params)
        teacher_cal = teacher_cal.astype(np.float32)
        student_cal = student_cal.astype(np.float32)
        with np.load(cal_dir / "selected_predictions.npz", allow_pickle=False) as a:
            prior_student = np.asarray(a["test_prediction"], dtype=np.float32)
            prior_truth = np.asarray(a["test_truth"], dtype=np.float32)
            prior_endpoints = np.asarray(a["test_endpoints"], dtype=np.int64)
        if not (np.array_equal(prior_student, student_cal)
                and np.array_equal(prior_truth, test_truth.astype(np.float32))
                and np.array_equal(prior_endpoints, test_endpoints)):
            raise AssertionError(f"{dataset}: existing Student ValCal Test artifact not reproduced")
        ci = paired_ci(teacher_cal, student_cal, test_truth, test_endpoints, dataset)
        row = {
            "teacher_source": str(teacher_source.resolve()),
            "student_raw_source": str(student_raw_path.resolve()),
            "student_calibrated_source": str((cal_dir / "selected_predictions.npz").resolve()),
            "params": params,
            "val_samples": len(val_truth),
            "test_samples": len(test_truth),
            "teacher_cache_padding_discarded": {"val": val_padding, "test": test_padding},
            "student_existing_calibrated_test_exactly_reproduced": True,
            "teacher": {
                "raw_test": metrics(teacher_test, test_truth),
                "matched_calibrated_test": metrics(teacher_cal, test_truth),
                "table_values": int(sum(t.size for t in teacher_tables)),
            },
            "student": {
                "raw_test": metrics(student_test, test_truth),
                "matched_calibrated_test": metrics(student_cal, test_truth),
                "table_values": int(sum(t.size for t in student_tables)),
            },
            "paired_bootstrap": ci,
        }
        np.savez_compressed(OUT / f"{dataset.lower()}_teacher_matched_predictions.npz",
                            test_prediction=teacher_cal,
                            test_truth=test_truth.astype(np.float32),
                            test_endpoints=test_endpoints)
        (OUT / f"{dataset.lower()}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        report["datasets"][dataset] = row
        print(f"{dataset}: Teacher {row['teacher']['matched_calibrated_test']['mae']:.6f}, "
              f"Student {row['student']['matched_calibrated_test']['mae']:.6f}", flush=True)
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
