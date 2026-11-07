"""Complete the existing matched Teacher/Student ValCal protocol on PEMS03/08."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from calibrate_dualscale_quantile_residual import metrics
from calibrate_dualscale_smooth_tod import fit_and_apply
from run_e1_e3 import teacher
from run_matched_teacher_test import paired_ci

OUT = HERE / "matched_teacher_test_pems03_08_20261024"
SOURCES = {
    "PEMS03": (
        "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal",
        "datasets/PEMS03/PEMS03_12to12_parent_predictions_joint3.npz"),
    "PEMS08": (
        "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal",
        "datasets/PEMS08/PEMS08_12to12_parent_predictions_joint3.npz"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    report = dict(protocol="existing_E1_equal_ValCal_fair_Test_extended_to_03_08",
                  test_tuning=False, dataset_rows={})
    for dataset, (cal_rel, teacher_rel) in SOURCES.items():
        cal_dir = ROOT / cal_rel
        config = json.loads((cal_dir / "calibration.json").read_text(encoding="utf-8"))
        params = {key: config[key] for key in (
            "num_bins", "smooth_shrinkage", "num_time_bins",
            "time_shrinkage", "slices_per_day")}
        raw_path = Path(config["source_predictions"])
        if not raw_path.exists():
            raw_path = ROOT / raw_path
        with np.load(raw_path, allow_pickle=False) as archive:
            student_val = np.asarray(archive["val_prediction"], dtype=np.float64)
            val_truth = np.asarray(archive["val_truth"], dtype=np.float64)
            val_endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
            student_test = np.asarray(archive["test_prediction"], dtype=np.float64)
            test_truth = np.asarray(archive["test_truth"], dtype=np.float64)
            test_endpoints = np.asarray(archive["test_endpoints"], dtype=np.int64)
        teacher_path = ROOT / teacher_rel
        teacher_val, val_padding = teacher(teacher_path, "val", len(val_truth))
        teacher_test, test_padding = teacher(teacher_path, "test", len(test_truth))
        if not (student_val.shape == teacher_val.shape == val_truth.shape and
                student_test.shape == teacher_test.shape == test_truth.shape):
            raise ValueError(f"{dataset}: prediction/truth shapes differ")
        if not (np.all(np.diff(val_endpoints) > 0) and
                np.all(np.diff(test_endpoints) > 0)):
            raise ValueError(f"{dataset}: endpoints not monotonic")
        _, teacher_cal, teacher_tables = fit_and_apply(
            teacher_val.astype(np.float64), val_truth, val_endpoints,
            teacher_test.astype(np.float64), test_endpoints, **params)
        _, student_cal, student_tables = fit_and_apply(
            student_val, val_truth, val_endpoints,
            student_test, test_endpoints, **params)
        teacher_cal = teacher_cal.astype(np.float32)
        student_cal = student_cal.astype(np.float32)
        with np.load(cal_dir / "selected_predictions.npz", allow_pickle=False) as archive:
            old_student = np.asarray(archive["test_prediction"], dtype=np.float32)
            old_truth = np.asarray(archive["test_truth"], dtype=np.float32)
            old_endpoints = np.asarray(archive["test_endpoints"], dtype=np.int64)
        if not (np.array_equal(old_student, student_cal) and
                np.array_equal(old_truth, test_truth.astype(np.float32)) and
                np.array_equal(old_endpoints, test_endpoints)):
            raise AssertionError(f"{dataset}: existing Student calibrated output not reproduced")
        row = dict(dataset=dataset, teacher_source=str(teacher_path),
                   student_raw_source=str(raw_path), calibration_params=params,
                   val_samples=len(val_truth), test_samples=len(test_truth),
                   teacher_padding_discarded=dict(val=val_padding, test=test_padding),
                   student_prior_exactly_reproduced=True,
                   teacher=dict(raw_test=metrics(teacher_test, test_truth),
                                matched_calibrated_test=metrics(teacher_cal, test_truth),
                                table_values=int(sum(t.size for t in teacher_tables))),
                   student=dict(raw_test=metrics(student_test, test_truth),
                                matched_calibrated_test=metrics(student_cal, test_truth),
                                table_values=int(sum(t.size for t in student_tables))),
                   paired_bootstrap=paired_ci(teacher_cal, student_cal, test_truth,
                                              test_endpoints, dataset))
        np.savez_compressed(OUT / f"{dataset.lower()}_teacher_matched_predictions.npz",
                            test_prediction=teacher_cal,
                            test_truth=test_truth.astype(np.float32),
                            test_endpoints=test_endpoints)
        (OUT / f"{dataset.lower()}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        report["dataset_rows"][dataset] = row
        print(dataset, "Teacher", row["teacher"]["matched_calibrated_test"]["mae"],
              "Student", row["student"]["matched_calibrated_test"]["mae"], flush=True)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
