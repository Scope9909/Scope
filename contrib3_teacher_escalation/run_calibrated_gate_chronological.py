"""Chronological Validation-only calibrated Student/Teacher utility-gate screen.

Four disjoint phases: calibrator fit, gate fit, threshold selection, evaluation.
No Test keys are accessed. Saved ValCal hyperparameters were chosen on Validation
previously, so this is a feasibility screen, not independent confirmation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT_PATH = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_PATH))

from run_pems07_teacher_escalation import (
    BUDGETS, EMBARGO, SEEDS, build_features, evaluate_fixed_thresholds,
    evaluate_method, evaluate_safe_budget_thresholds, routed_mae, split_utility,
    strict_online_indices, train_one,
)
import run_pems07_teacher_escalation as routing_impl
from screen_matched_valcal_oracle import CALIBRATION_KEYS, ROOT, SOURCES, fit_and_apply, official_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(SOURCES), required=True)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--export-gate", action="store_true")
    parser.add_argument("--align-formal-pems04", action="store_true",
                        help="Use the published 17.5064 PEMS04 Student and full-Validation Test calibrator")
    parser.add_argument("--align-formal-all", action="store_true",
                        help="Use each dataset's published Student and full-Validation Test calibrator")
    parser.add_argument("--student-seed", type=int, choices=(1, 2, 3), default=1,
                        help="Select the matched frozen Student/checkpoint calibration seed")
    parser.add_argument("--pems04-strictalign", action="store_true",
                        help="Use the matched PEMS04 strict-align checkpoint family for seed comparisons")
    parser.add_argument("--explore-17-19", action="store_true",
                        help="Post-hoc 17/19 percent budget audit; retain 10/20 percent parity controls")
    args = parser.parse_args()
    if args.explore_17_19:
        routing_impl.BUDGETS = (0.10, 0.17, 0.19, 0.20)
    dataset = args.dataset
    if args.align_formal_pems04 and dataset != "PEMS04":
        parser.error("--align-formal-pems04 requires --dataset PEMS04")
    if args.student_seed != 1 and not args.align_formal_all:
        parser.error("Student seeds 2/3 require --align-formal-all")
    if args.pems04_strictalign and (dataset != "PEMS04" or not args.align_formal_all):
        parser.error("--pems04-strictalign requires PEMS04 and --align-formal-all")
    align_formal = args.align_formal_pems04 or args.align_formal_all
    paths = dict(SOURCES[dataset])
    suffix = ("_formal17p5064" if dataset == "PEMS04" else "_formal") if align_formal else ""
    if args.student_seed != 1:
        seed_token = f"s{args.student_seed}"
        if dataset == "PEMS04":
            paths["student"] = f"paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_{seed_token}/selected_predictions.npz"
            paths["calibration"] = f"paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_{seed_token}_valcal/calibration.json"
            suffix = "_formal_strictalign"
        else:
            paths["student"] = paths["student"].replace("s1_network_only", f"{seed_token}_network_only")
            paths["calibration"] = paths["calibration"].replace("s1_valcal", f"{seed_token}_valcal")
            if paths["student"] == SOURCES[dataset]["student"] or paths["calibration"] == SOURCES[dataset]["calibration"]:
                raise ValueError("Student/calibration seed path substitution failed")
        suffix += f"_s{args.student_seed}"
    if align_formal and dataset == "PEMS04":
        if args.pems04_strictalign or args.student_seed != 1:
            seed_token = f"s{args.student_seed}"
            paths["student"] = f"paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_{seed_token}/selected_predictions.npz"
            paths["calibration"] = f"paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_{seed_token}_valcal/calibration.json"
            suffix = f"_formal_strictalign_{seed_token}"
        else:
            paths["student"] = "paper_results/lite_r256_kdfix3_a30_lr1e4_aligned_s1/selected_predictions.npz"
            paths["calibration"] = "paper_results/lite_r256_kdfix3_a30_lr1e4_smooth_tod_s1/calibration.json"
    if args.explore_17_19:
        suffix += "_budget17_19"
    with np.load(ROOT / paths["student"], allow_pickle=False) as archive:
        endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
        student = np.asarray(archive["val_prediction"], dtype=np.float32)
        truth = np.asarray(archive["val_truth"], dtype=np.float32)
        if args.final_test:
            test_endpoints = np.asarray(archive["test_endpoints"], dtype=np.int64)
            test_student = np.asarray(archive["test_prediction"], dtype=np.float32)
            test_truth = np.asarray(archive["test_truth"], dtype=np.float32)
    with np.load(ROOT / paths["teacher"], allow_pickle=False) as archive:
        teacher = np.asarray(archive["teacher_pred_val"], dtype=np.float32)[..., 0]
        scaler_std, scaler_mean = float(archive["scaler_std"]), float(archive["scaler_mean"])
        teacher = teacher * scaler_std + scaler_mean
        if args.final_test:
            test_teacher = np.asarray(archive["teacher_pred_test"][:len(test_endpoints)], dtype=np.float32)[..., 0]
            test_teacher = test_teacher * scaler_std + scaler_mean
    with np.load(ROOT / f"datasets/{dataset}/{dataset}_12to12_index_scaler.npz", allow_pickle=False) as archive:
        official_endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
        if args.final_test:
            official_test_endpoints = np.asarray(archive["idx_test"], dtype=np.int64)
    if not np.array_equal(endpoints, official_endpoints):
        raise ValueError("Validation endpoints mismatch")
    if not (student.shape == teacher.shape == truth.shape):
        raise ValueError("Student, Teacher, and truth shapes mismatch")
    if args.final_test and (
        not np.array_equal(test_endpoints, official_test_endpoints)
        or not (test_student.shape == test_teacher.shape == test_truth.shape)
    ):
        raise ValueError("Test endpoints or prediction shapes mismatch")
    n = len(endpoints)
    q1, q2, q3 = n // 4, n // 2, 3 * n // 4
    cal = slice(0, q1)
    gate = slice(q1 + EMBARGO, q2)
    selection = slice(q2 + EMBARGO, q3)
    evaluation = slice(q3 + EMBARGO, n)
    if min(gate.stop-gate.start, selection.stop-selection.start, evaluation.stop-evaluation.start) < 100:
        raise ValueError("Too few windows in one chronological phase")
    config = json.loads((ROOT / paths["calibration"]).read_text(encoding="utf-8"))
    params = {key: config[key] for key in CALIBRATION_KEYS}
    post = slice(gate.start, n)
    print(dataset, "calibrating Student", flush=True)
    _, student_cal, _ = fit_and_apply(
        student[cal].astype(np.float64), truth[cal].astype(np.float64), endpoints[cal],
        student[post].astype(np.float64), endpoints[post], **params,
    )
    print(dataset, "calibrating Teacher", flush=True)
    _, teacher_cal, _ = fit_and_apply(
        teacher[cal].astype(np.float64), truth[cal].astype(np.float64), endpoints[cal],
        teacher[post].astype(np.float64), endpoints[post], **params,
    )
    del student, teacher
    with np.load(ROOT / f"datasets/{dataset}/data.npz", allow_pickle=False) as archive:
        raw = np.asarray(archive["data"], dtype=np.float32)[..., 0]
    print(dataset, "building visible features", flush=True)
    features, volatility = build_features(raw, endpoints[post], student_cal.astype(np.float32))
    del raw
    prediction_mean = student_cal.mean(axis=(1, 2))
    valid = np.isfinite(truth[post]) & (np.rint(truth[post]) > 0)
    student_error, count = official_error(student_cal, truth[post])
    teacher_error, teacher_count = official_error(teacher_cal, truth[post])
    if not np.array_equal(count, teacher_count):
        raise ValueError("Validity masks differ")
    del valid, student_cal, teacher_cal, truth
    def loc(part):
        return np.arange(part.start-post.start, part.stop-post.start)
    gi, si, ei = loc(gate), loc(selection), loc(evaluation)
    feature_mean = features[gi].mean(axis=0, dtype=np.float64)
    feature_std = features[gi].std(axis=0, dtype=np.float64)
    feature_std[feature_std < 1e-6] = 1.0
    x = ((features-feature_mean)/feature_std).astype(np.float32)
    gate_u = split_utility(student_error[gi], teacher_error[gi], count[gi])
    inner_end = gate.start + 3*(gate.stop-gate.start)//4
    inner_train = np.arange(gate.start-post.start, inner_end-post.start)
    inner_val = np.arange(inner_end+EMBARGO-post.start, gate.stop-post.start)
    inner_u_train = split_utility(student_error[inner_train], teacher_error[inner_train], count[inner_train])
    inner_u_val = split_utility(student_error[inner_val], teacher_error[inner_val], count[inner_val])
    def scale(values, mean, std):
        return ((values-mean)/max(std, 1e-8)).astype(np.float32)
    y_gate = scale(gate_u, gate_u.mean(), gate_u.std())
    y_inner_train = scale(inner_u_train, inner_u_train.mean(), inner_u_train.std())
    y_inner_val = scale(inner_u_val, inner_u_train.mean(), inner_u_train.std())
    training = []
    models = []
    for seed in SEEDS:
        model, record = train_one(seed, x[inner_train], y_inner_train,
                                  x[inner_val], y_inner_val, x[gi], y_gate)
        models.append(model)
        training.append(record)
        print(dataset, "seed", seed, "epoch", record["best_epoch"], flush=True)
    def scores(indices):
        tensor = torch.from_numpy(x[indices])
        with torch.no_grad():
            return np.mean([model.eval()(tensor).numpy() for model in models], axis=0)
    selection_gate = scores(si)
    evaluation_gate = scores(ei)
    if args.export_gate:
        gate_out = HERE / f"{dataset.lower()}{suffix}_calibrated_utility_gate_frozen.pt"
        torch.save({
            "protocol": "validation_4_chronological_phases_gate_fixed_before_test",
            "dataset": dataset,
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "selection_threshold_10pct": float(np.quantile(selection_gate, 0.9)),
            "budget": 0.1,
            "model_states": [model.state_dict() for model in models],
            "validation_training_records": training,
        }, gate_out)
        print(gate_out, flush=True)
    eval_student, eval_teacher, eval_count = student_error[ei], teacher_error[ei], count[ei]
    selection_utility = split_utility(student_error[si], teacher_error[si], count[si])
    eval_utility = split_utility(eval_student, eval_teacher, eval_count)
    baseline = routed_mae(eval_student, eval_teacher, eval_count, np.empty(0, dtype=np.int64))
    teacher_mae = routed_mae(eval_student, eval_teacher, eval_count, np.arange(len(ei)))
    arms = {
        "calibrated_utility_gate": (selection_gate, evaluation_gate),
        "historical_volatility": (volatility[si], volatility[ei]),
        "calibrated_student_prediction_mean": (prediction_mean[si], prediction_mean[ei]),
    }
    results = {}
    for name, (selection_score, eval_score) in arms.items():
        results[name] = {
            "strict_online": evaluate_fixed_thresholds(selection_score, eval_score,
                                                        eval_student, eval_teacher, eval_count),
            "offline_top_k": evaluate_method(eval_score, eval_student, eval_teacher, eval_count),
        }
    results["calibrated_utility_gate"]["safe_abstain"] = evaluate_safe_budget_thresholds(
        selection_gate, selection_utility, evaluation_gate,
        eval_student, eval_teacher, eval_count,
    )
    results["oracle"] = {"offline_top_k": evaluate_method(eval_utility, eval_student, eval_teacher, eval_count)}
    report = {
        "dataset": dataset,
        "student_seed": args.student_seed,
        "protocol": "validation_4_chronological_phases_cal_fit_gate_fit_threshold_selection_final_evaluation",
        "test_keys_read": False,
        "caveat": "ValCal hyperparameters were selected on Validation previously; this is not independent final confirmation.",
        "metric": "mean of 12 masked horizon MAEs",
        "calibration": [cal.start, cal.stop-1],
        "gate_fit": [gate.start, gate.stop-1],
        "inner_train": [int(inner_train[0]+post.start), int(inner_train[-1]+post.start)],
        "inner_validation": [int(inner_val[0]+post.start), int(inner_val[-1]+post.start)],
        "threshold_selection": [selection.start, selection.stop-1],
        "evaluation": [evaluation.start, evaluation.stop-1],
        "student_mae": baseline,
        "teacher_mae": teacher_mae,
        "teacher_always_gain": baseline-teacher_mae,
        "teacher_better_fraction": float(np.mean(eval_utility > 0)),
        "training": training,
        "methods": results,
    }
    out = HERE / f"{dataset.lower()}{suffix}_calibrated_gate_chronological.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(dataset, "student", baseline, "teacher", teacher_mae, flush=True)
    for name, result in results.items():
        arm = result.get("strict_online", result["offline_top_k"])["0.1"]
        print(dataset, name, arm["gain_vs_student"], arm.get("realized_evaluation_budget"), flush=True)
    print(out, flush=True)
    if args.final_test:
        # All model weights, score normalization, and thresholds were fixed above
        # using Validation only. Test enters only this final evaluation block.
        test_cal_stop = n if align_formal else q1
        test_cal = slice(0, test_cal_stop)
        with np.load(ROOT / paths["student"], allow_pickle=False) as archive:
            val_student_cal = np.asarray(archive["val_prediction"][test_cal], dtype=np.float64)
            val_truth_cal = np.asarray(archive["val_truth"][test_cal], dtype=np.float64)
        _, student_test_cal, _ = fit_and_apply(
            val_student_cal, val_truth_cal, endpoints[test_cal],
            test_student.astype(np.float64), test_endpoints, **params,
        )
        with np.load(ROOT / paths["teacher"], allow_pickle=False) as archive:
            val_teacher_cal = np.asarray(archive["teacher_pred_val"][test_cal], dtype=np.float64)[..., 0]
            val_teacher_cal = val_teacher_cal * scaler_std + scaler_mean
        _, teacher_test_cal, _ = fit_and_apply(
            val_teacher_cal, val_truth_cal, endpoints[test_cal],
            test_teacher.astype(np.float64), test_endpoints, **params,
        )
        with np.load(ROOT / f"datasets/{dataset}/data.npz", allow_pickle=False) as archive:
            raw_test = np.asarray(archive["data"], dtype=np.float32)[..., 0]
        test_features, test_volatility = build_features(
            raw_test, test_endpoints, student_test_cal.astype(np.float32)
        )
        test_x = ((test_features-feature_mean)/feature_std).astype(np.float32)
        with torch.no_grad():
            test_gate = np.mean([
                model.eval()(torch.from_numpy(test_x)).numpy() for model in models
            ], axis=0)
        test_mean = student_test_cal.mean(axis=(1, 2))
        test_student_error, test_count = official_error(student_test_cal, test_truth)
        test_teacher_error, test_teacher_count = official_error(teacher_test_cal, test_truth)
        if not np.array_equal(test_count, test_teacher_count):
            raise ValueError("Test validity masks differ")
        test_baseline = routed_mae(test_student_error, test_teacher_error, test_count,
                                   np.empty(0, dtype=np.int64))
        if align_formal:
            expected_formal = float(config["calibrated_test"]["mae"])
            if abs(test_baseline - expected_formal) > 0.002:
                raise ValueError(f"Formal {dataset} baseline did not reproduce: {test_baseline} vs {expected_formal}")
        test_teacher_mae = routed_mae(test_student_error, test_teacher_error, test_count,
                                      np.arange(len(test_endpoints)))
        test_arms = {
            "calibrated_utility_gate": (selection_gate, test_gate),
            "historical_volatility": (volatility[si], test_volatility),
            "calibrated_student_prediction_mean": (prediction_mean[si], test_mean),
        }
        test_results = {}
        for name, (selection_score, test_score) in test_arms.items():
            test_results[name] = evaluate_fixed_thresholds(
                selection_score, test_score,
                test_student_error, test_teacher_error, test_count,
            )
        test_results["calibrated_utility_gate_safe_abstain"] = evaluate_safe_budget_thresholds(
            selection_gate, selection_utility, test_gate,
            test_student_error, test_teacher_error, test_count,
        )
        final = {
            "dataset": dataset,
            "student_seed": args.student_seed,
            "protocol": "frozen_validation_calibration_gate_threshold_evaluated_once_on_test",
            "alignment": "formal_Student_full_Validation_Test_calibration" if align_formal else "original_first_quarter_Validation_Test_calibration",
            "caveat_alignment": "Gate and threshold are trained on early-Validation-calibrated predictions; Test uses the formal full-Validation calibrator, so score distribution can shift. This is a diagnostic, not a globally nested independent evaluation." if align_formal else None,
            "test_endpoints": [int(test_endpoints[0]), int(test_endpoints[-1])],
            "test_windows": int(len(test_endpoints)),
            "validation_protocol_file": str(out.name),
            "student_mae": test_baseline,
            "teacher_mae": test_teacher_mae,
            "teacher_always_gain": test_baseline-test_teacher_mae,
            "teacher_better_fraction": float(np.mean(split_utility(
                test_student_error, test_teacher_error, test_count
            ) > 0)),
            "methods": test_results,
            "caveat": "Teacher test cache prefix alignment follows official idx_test and prior project cache convention; no Test values were used for tuning.",
        }
        test_out = HERE / f"{dataset.lower()}{suffix}_calibrated_gate_final_test.json"
        test_out.write_text(json.dumps(final, indent=2), encoding="utf-8")
        if dataset == "PEMS07":
            chosen = {}
            for name, (selection_score, test_score) in test_arms.items():
                threshold = float(np.quantile(selection_score, 0.9))
                selected = strict_online_indices(test_score, threshold, 0.1)
                flags = np.zeros(len(test_endpoints), dtype=np.bool_)
                flags[selected] = True
                chosen[name] = flags
            audit_out = HERE / f"pems07{suffix}_calibrated_gate_final_test_paired_audit.npz"
            np.savez_compressed(
                audit_out,
                endpoints=test_endpoints,
                student_error_by_horizon=test_student_error,
                teacher_error_by_horizon=test_teacher_error,
                valid_count_by_horizon=test_count,
                gate_score=test_gate,
                volatility_score=test_volatility,
                student_mean_score=test_mean,
                gate_selected=chosen["calibrated_utility_gate"],
                volatility_selected=chosen["historical_volatility"],
                student_mean_selected=chosen["calibrated_student_prediction_mean"],
            )
            print(audit_out, flush=True)
        print(dataset, "TEST student", test_baseline, "teacher", test_teacher_mae, flush=True)
        for name, result in test_results.items():
            arm = result["0.1"]
            print(dataset, "TEST", name, arm["gain_vs_student"],
                  arm["realized_evaluation_budget"], flush=True)
        print(test_out, flush=True)


if __name__ == "__main__":
    main()
