"""Validation-only transfer screen of the frozen PEMS07 utility scores to matched ValCal.

This does not refit the gate. It tests whether its saved ranking remains useful
after fitting the same calibration recipe to Student and Teacher separately.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT_PATH = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_PATH))

from run_pems07_teacher_escalation import (
    evaluate_fixed_thresholds,
    evaluate_method,
    routed_mae,
    split_utility,
)
from screen_matched_valcal_oracle import CALIBRATION_KEYS, ROOT, SOURCES, fit_and_apply, official_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(SOURCES), default="PEMS07")
    args = parser.parse_args()
    dataset = args.dataset
    paths = SOURCES[dataset]
    with np.load(ROOT / paths["student"], allow_pickle=False) as archive:
        endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
        student = np.asarray(archive["val_prediction"], dtype=np.float32)
        truth = np.asarray(archive["val_truth"], dtype=np.float32)
    with np.load(ROOT / paths["teacher"], allow_pickle=False) as archive:
        teacher = np.asarray(archive["teacher_pred_val"], dtype=np.float32)[..., 0]
        teacher = teacher * float(archive["scaler_std"]) + float(archive["scaler_mean"])
    with np.load(HERE / f"{dataset.lower()}_teacher_escalation_scores.npz", allow_pickle=False) as archive:
        eval_indices = np.asarray(archive["evaluation_indices"], dtype=np.int64)
        eval_endpoints = np.asarray(archive["evaluation_endpoints"], dtype=np.int64)
        gate_eval = np.asarray(archive["learned_gate_score"], dtype=np.float64)
        gate_selection = np.asarray(archive["selection_gate_score"], dtype=np.float64)
        volatility_eval = np.asarray(archive["historical_volatility_score"], dtype=np.float64)
    n = len(endpoints)
    fit = slice(0, n // 2)
    selection = slice(n // 2 + 12, 3 * n // 4)
    evaluation = slice(3 * n // 4 + 12, n)
    if not np.array_equal(eval_indices, np.arange(evaluation.start, evaluation.stop)):
        raise ValueError("Saved gate indices do not match the calibrated evaluation slice")
    if not np.array_equal(eval_endpoints, endpoints[evaluation]):
        raise ValueError("Saved gate endpoints do not match the Student cache")
    if len(gate_selection) != selection.stop - selection.start:
        raise ValueError("Saved selection gate length does not match")
    if not (student.shape == teacher.shape == truth.shape):
        raise ValueError("Student, Teacher and truth are misaligned")
    config = json.loads((ROOT / paths["calibration"]).read_text(encoding="utf-8"))
    params = {key: config[key] for key in CALIBRATION_KEYS}
    post = slice(selection.start, n)
    _, student_cal, _ = fit_and_apply(
        student[fit].astype(np.float64), truth[fit].astype(np.float64), endpoints[fit],
        student[post].astype(np.float64), endpoints[post], **params,
    )
    _, teacher_cal, _ = fit_and_apply(
        teacher[fit].astype(np.float64), truth[fit].astype(np.float64), endpoints[fit],
        teacher[post].astype(np.float64), endpoints[post], **params,
    )
    offset = evaluation.start - post.start
    student_sel, student_eval = student_cal[:offset][:len(gate_selection)], student_cal[offset:]
    teacher_sel, teacher_eval = teacher_cal[:offset][:len(gate_selection)], teacher_cal[offset:]
    selection_truth, eval_truth = truth[selection], truth[evaluation]
    student_sel_err, sel_count = official_error(student_sel, selection_truth)
    teacher_sel_err, teacher_sel_count = official_error(teacher_sel, selection_truth)
    student_err, count = official_error(student_eval, eval_truth)
    teacher_err, teacher_count = official_error(teacher_eval, eval_truth)
    if not (np.array_equal(sel_count, teacher_sel_count) and np.array_equal(count, teacher_count)):
        raise ValueError("Validity masks differ")
    utility_selection = split_utility(student_sel_err, teacher_sel_err, sel_count)
    utility_eval = split_utility(student_err, teacher_err, count)
    student_mae = routed_mae(student_err, teacher_err, count, np.empty(0, dtype=np.int64))
    teacher_mae = routed_mae(student_err, teacher_err, count, np.arange(len(count)))
    means_selection = student_sel.mean(axis=(1, 2))
    means_eval = student_eval.mean(axis=(1, 2))
    with np.load(ROOT / f"datasets/{dataset}/data.npz", allow_pickle=False) as archive:
        raw = np.asarray(archive["data"], dtype=np.float32)[..., 0]
    volatility_selection = np.asarray([
        np.abs(np.diff(raw[int(ep)-11:int(ep)+1], axis=0)).mean()
        for ep in endpoints[selection]
    ], dtype=np.float64)
    methods = {
        "frozen_old_utility_gate": (gate_selection, gate_eval),
        "historical_volatility": (volatility_selection, volatility_eval),
        "calibrated_student_prediction_mean": (means_selection, means_eval),
    }
    results = {}
    for name, (select_score, eval_score) in methods.items():
        results[name] = {
            "offline_top_k": evaluate_method(eval_score, student_err, teacher_err, count),
            "strict_online": evaluate_fixed_thresholds(
                select_score, eval_score, student_err, teacher_err, count
            ),
        }
    results["oracle"] = {"offline_top_k": evaluate_method(
        utility_eval, student_err, teacher_err, count
    )}
    report = {
        "dataset": dataset,
        "protocol": "validation_only_frozen_old_gate_matched_ValCal_screen",
        "test_keys_read": False,
        "caveat": "Old gate scores were trained on uncalibrated utility and old Student forecasts; no gate refit. ValCal hyperparameters were selected on Validation previously. This is a transfer feasibility screen, not final confirmation.",
        "fit_indices": [0, fit.stop-1],
        "selection_indices": [selection.start, selection.stop-1],
        "evaluation_indices": [evaluation.start, evaluation.stop-1],
        "student_mae": student_mae,
        "teacher_mae": teacher_mae,
        "teacher_always_gain": student_mae-teacher_mae,
        "teacher_better_window_fraction": float(np.mean(utility_eval > 0)),
        "methods": results,
    }
    out = HERE / f"{dataset.lower()}_frozen_gate_matched_valcal_screen.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("student", student_mae, "teacher", teacher_mae)
    for name, result in results.items():
        part = result.get("strict_online", result["offline_top_k"])["0.1"]
        print(name, part["gain_vs_student"], part.get("realized_evaluation_budget"))
    print(out)


if __name__ == "__main__":
    main()
