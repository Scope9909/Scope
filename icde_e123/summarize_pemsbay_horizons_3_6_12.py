"""Summarize exact forecast leads 3/6/12 from existing PEMS-BAY outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from calibrate_dualscale_smooth_tod import fit_and_apply
from run_e1_e3 import teacher

CAL_DIR = ROOT / "paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal"
TEACHER = ROOT / "datasets/PEMSBAY/PEMSBAY_12to12_parent_predictions_final_joint_712.npz"
OUT = HERE / "pemsbay_horizons_3_6_12.json"


def lead_mae(prediction, truth, lead):
    p = prediction[:, lead - 1]
    y = truth[:, lead - 1]
    valid = np.rint(y) > 0
    return float(np.mean(np.abs(p[valid] - y[valid])))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    config = json.loads((CAL_DIR / "calibration.json").read_text(encoding="utf-8"))
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
    teacher_val, val_padding = teacher(TEACHER, "val", len(val_truth))
    teacher_test, test_padding = teacher(TEACHER, "test", len(test_truth))
    if not (student_val.shape == teacher_val.shape == val_truth.shape and
            student_test.shape == teacher_test.shape == test_truth.shape):
        raise ValueError("PEMS-BAY shape mismatch")
    _, teacher_cal, _ = fit_and_apply(
        teacher_val.astype(np.float64), val_truth, val_endpoints,
        teacher_test.astype(np.float64), test_endpoints, **params)
    _, student_cal, _ = fit_and_apply(
        student_val, val_truth, val_endpoints,
        student_test, test_endpoints, **params)
    with np.load(CAL_DIR / "selected_predictions.npz", allow_pickle=False) as archive:
        prior = np.asarray(archive["test_prediction"], dtype=np.float32)
    if not np.array_equal(prior, student_cal.astype(np.float32)):
        raise AssertionError("Existing calibrated Student cannot be reproduced")
    rows = []
    for lead in (3, 6, 12):
        row = dict(lead=lead,
                   teacher_raw_mae=lead_mae(teacher_test, test_truth, lead),
                   teacher_matched_calibrated_mae=lead_mae(teacher_cal, test_truth, lead),
                   student_raw_mae=lead_mae(student_test, test_truth, lead),
                   student_calibrated_mae=lead_mae(student_cal, test_truth, lead))
        row["teacher_cal_minus_student_cal"] = (
            row["teacher_matched_calibrated_mae"] - row["student_calibrated_mae"])
        rows.append(row)
    OUT.write_text(json.dumps(dict(dataset="PEMSBAY", role="3/6/12-step accuracy only",
                                   codec_claim=False, lead_definition="individual horizon index, not 1..h mean",
                                   model_seed=1, test_samples=len(test_truth),
                                   teacher_padding_discarded=dict(val=val_padding, test=test_padding),
                                   calibration_params=params,
                                   student_prior_exactly_reproduced=True,
                                   rows=rows), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
