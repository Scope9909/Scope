"""PEMS07 validation-only pilot for budgeted Student-to-Teacher escalation.

The Student and Teacher are frozen.  A small utility gate predicts, from
inference-visible history and Student forecasts, which windows benefit most
from paying for the Teacher.  No Test key is read by this script.
"""

from __future__ import annotations

import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

BUDGETS = (0.05, 0.10, 0.20)
SEEDS = (17, 29, 43, 71, 101)
EMBARGO = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def summarize_sequence(sequence: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    """Summarize [sample, time, node] without retaining a large copy."""
    rows = []
    for start in range(0, len(sequence), chunk_size):
        x = np.asarray(sequence[start : start + chunk_size], dtype=np.float32)
        q = np.quantile(x, (0.1, 0.5, 0.9), axis=2).transpose(1, 2, 0)
        level = np.concatenate(
            (x.mean(axis=2, keepdims=True), x.std(axis=2, keepdims=True), q),
            axis=2,
        ).reshape(len(x), -1)
        delta = np.diff(x, axis=1)
        delta_stats = np.stack(
            (
                delta.mean(axis=2),
                delta.std(axis=2),
                np.abs(delta).mean(axis=2),
                np.quantile(np.abs(delta), 0.9, axis=2),
            ),
            axis=2,
        ).reshape(len(x), -1)
        overall = np.stack(
            (
                x.mean(axis=(1, 2)),
                x.std(axis=(1, 2)),
                np.quantile(x, 0.1, axis=(1, 2)),
                np.quantile(x, 0.9, axis=(1, 2)),
                np.abs(delta).mean(axis=(1, 2)),
                np.quantile(np.abs(delta), 0.9, axis=(1, 2)),
            ),
            axis=1,
        )
        rows.append(np.concatenate((level, delta_stats, overall), axis=1))
    return np.concatenate(rows, axis=0).astype(np.float32)


def build_features(
    raw: np.ndarray, endpoints: np.ndarray, student_prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    history = np.empty((len(endpoints), 12, raw.shape[1]), dtype=np.float32)
    for i, endpoint in enumerate(endpoints):
        history[i] = raw[int(endpoint) - 11 : int(endpoint) + 1]
    history_feature = summarize_sequence(history)
    volatility = np.abs(np.diff(history, axis=1)).mean(axis=(1, 2)).astype(np.float64)
    del history
    prediction_feature = summarize_sequence(student_prediction)
    step_of_day = (endpoints % 288).astype(np.float64)
    day_of_week = ((endpoints // 288) % 7).astype(np.float64)
    time_feature = np.stack(
        (
            np.sin(2.0 * np.pi * step_of_day / 288.0),
            np.cos(2.0 * np.pi * step_of_day / 288.0),
            np.sin(2.0 * np.pi * day_of_week / 7.0),
            np.cos(2.0 * np.pi * day_of_week / 7.0),
        ),
        axis=1,
    ).astype(np.float32)
    return np.concatenate((history_feature, prediction_feature, time_feature), axis=1), volatility


class UtilityGate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def top_indices(score: np.ndarray, budget: float) -> np.ndarray:
    count = max(1, int(math.floor(budget * len(score))))
    return np.argsort(score, kind="mergesort")[-count:]


def routed_mae(
    student_error_by_horizon: np.ndarray,
    teacher_error_by_horizon: np.ndarray,
    valid_count_by_horizon: np.ndarray,
    selected: np.ndarray,
) -> float:
    """Official metric: masked MAE per horizon, then average horizons."""
    total = student_error_by_horizon.sum(axis=0, dtype=np.float64)
    if len(selected):
        total += (
            teacher_error_by_horizon[selected] - student_error_by_horizon[selected]
        ).sum(axis=0, dtype=np.float64)
    count = valid_count_by_horizon.sum(axis=0, dtype=np.float64)
    return float(np.mean(total / np.maximum(count, 1.0)))


def split_utility(
    student_error_by_horizon: np.ndarray,
    teacher_error_by_horizon: np.ndarray,
    valid_count_by_horizon: np.ndarray,
) -> np.ndarray:
    """Per-window additive contribution to official MAE gain, scaled by n."""
    denominator = valid_count_by_horizon.sum(axis=0, dtype=np.float64)
    contribution = (
        student_error_by_horizon - teacher_error_by_horizon
    ) / np.maximum(denominator[None, :], 1.0)
    return contribution.mean(axis=1) * len(contribution)


def routing_gain(
    score: np.ndarray,
    budget: float,
    student_error_sum: np.ndarray,
    teacher_error_sum: np.ndarray,
    valid_count: np.ndarray,
) -> float:
    baseline = routed_mae(
        student_error_sum, teacher_error_sum, valid_count, np.empty(0, dtype=np.int64)
    )
    routed = routed_mae(
        student_error_sum,
        teacher_error_sum,
        valid_count,
        top_indices(score, budget),
    )
    return baseline - routed


def train_one(
    seed: int,
    x_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    x_inner_validation: np.ndarray,
    y_inner_validation: np.ndarray,
    x_refit: np.ndarray,
    y_refit: np.ndarray,
) -> tuple[UtilityGate, dict]:
    set_seed(seed)
    model = UtilityGate(x_inner_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_function = nn.SmoothL1Loss(beta=0.5)
    x_tensor = torch.from_numpy(x_inner_train)
    y_tensor = torch.from_numpy(y_inner_train.astype(np.float32))
    x_validation_tensor = torch.from_numpy(x_inner_validation)
    y_validation_tensor = torch.from_numpy(y_inner_validation.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    batch_size = 256
    for epoch in range(1, 201):
        model.train()
        order = torch.randperm(len(x_tensor), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_tensor[index]), y_tensor[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(x_validation_tensor), y_validation_tensor).item()
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= 25:
            break
    # Refit from scratch on all outer-Fit samples for the internally chosen epoch.
    set_seed(seed)
    final_model = UtilityGate(x_refit.shape[1])
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-3)
    x_refit_tensor = torch.from_numpy(x_refit)
    y_refit_tensor = torch.from_numpy(y_refit.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(best_epoch):
        final_model.train()
        order = torch.randperm(len(x_refit_tensor), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(final_model(x_refit_tensor[index]), y_refit_tensor[index])
            loss.backward()
            optimizer.step()
    return final_model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "inner_validation_loss": best_loss,
        "inner_epochs_run": epoch,
    }


def evaluate_method(
    score: np.ndarray,
    student_sum: np.ndarray,
    teacher_sum: np.ndarray,
    valid_count: np.ndarray,
) -> dict:
    baseline = routed_mae(student_sum, teacher_sum, valid_count, np.empty(0, dtype=np.int64))
    result = {}
    for budget in BUDGETS:
        selected = top_indices(score, budget)
        mae = routed_mae(student_sum, teacher_sum, valid_count, selected)
        result[str(budget)] = {
            "teacher_windows": int(len(selected)),
            "mae": mae,
            "gain_vs_student": baseline - mae,
        }
    return result


def evaluate_fixed_thresholds(
    selection_score: np.ndarray,
    evaluation_score: np.ndarray,
    student_sum: np.ndarray,
    teacher_sum: np.ndarray,
    valid_count: np.ndarray,
) -> dict:
    """Choose a threshold on Selection, with a strict online quota controller."""
    baseline = routed_mae(student_sum, teacher_sum, valid_count, np.empty(0, dtype=np.int64))
    result = {}
    for budget in BUDGETS:
        threshold = float(np.quantile(selection_score, 1.0 - budget))
        selected = strict_online_indices(evaluation_score, threshold, budget)
        mae = routed_mae(student_sum, teacher_sum, valid_count, selected)
        result[str(budget)] = {
            "selection_target_budget": budget,
            "score_threshold": threshold,
            "teacher_windows": int(len(selected)),
            "realized_evaluation_budget": float(len(selected) / len(evaluation_score)),
            "mae": mae,
            "gain_vs_student": baseline - mae,
        }
    return result


def strict_online_indices(
    score: np.ndarray, threshold: float, budget: float
) -> np.ndarray:
    """Chronological token bucket whose prefix and final usage never exceed budget."""
    tokens = 0.0
    capacity = max(1.0, math.floor(budget * len(score)))
    selected = []
    for index, value in enumerate(score):
        tokens = min(capacity, tokens + budget)
        if value >= threshold and tokens + 1e-12 >= 1.0:
            selected.append(index)
            tokens -= 1.0
    return np.asarray(selected, dtype=np.int64)


def evaluate_safe_budget_thresholds(
    selection_score: np.ndarray,
    selection_utility: np.ndarray,
    evaluation_score: np.ndarray,
    student_sum: np.ndarray,
    teacher_sum: np.ndarray,
    valid_count: np.ndarray,
) -> dict:
    """Use at most the budget; Selection may choose to abstain completely."""
    baseline = routed_mae(student_sum, teacher_sum, valid_count, np.empty(0, dtype=np.int64))
    selection_order = np.argsort(selection_score, kind="mergesort")[::-1]
    cumulative_benefit = np.cumsum(selection_utility[selection_order]) / len(selection_utility)
    result = {}
    for budget in BUDGETS:
        maximum = max(1, int(math.floor(budget * len(selection_score))))
        candidates = np.concatenate(([0.0], cumulative_benefit[:maximum]))
        chosen_count = int(np.argmax(candidates))
        if chosen_count == 0:
            threshold = float("inf")
            selected = np.empty(0, dtype=np.int64)
        else:
            threshold = float(selection_score[selection_order[chosen_count - 1]])
            selected = strict_online_indices(evaluation_score, threshold, budget)
        mae = routed_mae(student_sum, teacher_sum, valid_count, selected)
        result[str(budget)] = {
            "maximum_budget": budget,
            "selection_chosen_windows": chosen_count,
            "selection_chosen_budget": float(chosen_count / len(selection_score)),
            "selection_gain": float(candidates[chosen_count]),
            "score_threshold": threshold if np.isfinite(threshold) else None,
            "teacher_windows": int(len(selected)),
            "realized_evaluation_budget": float(len(selected) / len(evaluation_score)),
            "mae": mae,
            "gain_vs_student": baseline - mae,
        }
    return result


def paths_for(dataset: str) -> dict:
    lower = dataset.lower()
    if dataset == "PEMS04":
        student = (
            ROOT
            / "paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_s1"
            / "selected_predictions.npz"
        )
        teacher = ROOT / "datasets/PEMS04/PEMS04_12to12_parent_predictions_joint3_strict.npz"
    else:
        student = (
            ROOT
            / f"paper_results/{lower}_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only"
            / "selected_predictions.npz"
        )
        teacher = ROOT / f"datasets/{dataset}/{dataset}_12to12_parent_predictions_joint3.npz"
    raw = ROOT / f"datasets/{dataset}/data.npz"
    risk = (
        ROOT
        / f"EXP_experiments/TLAST_improved/{dataset}"
        / f"TLAST_improved_{dataset}_12to12_expcontrib3_studentrisk_{lower}_h16_w100_s1"
        / "validation_output.npz"
    )
    return {
        "student": student,
        "teacher": teacher,
        "raw": raw,
        "risk": risk if risk.exists() else None,
        "result": OUT_DIR / f"{lower}_teacher_escalation_result.json",
        "scores": OUT_DIR / f"{lower}_teacher_escalation_scores.npz",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("PEMS03", "PEMS04", "PEMS07", "PEMS08"), default="PEMS07"
    )
    args = parser.parse_args()
    dataset = args.dataset
    paths = paths_for(dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Intentionally access only Validation keys from prediction archives.
    with np.load(paths["student"]) as archive:
        endpoints = archive["val_endpoints"].astype(np.int64)
        student = archive["val_prediction"].astype(np.float32)
        truth = archive["val_truth"].astype(np.float32)
    with np.load(paths["teacher"]) as archive:
        teacher_normalized = archive["teacher_pred_val"].astype(np.float32)[..., 0]
        scaler_mean = float(archive["scaler_mean"])
        scaler_std = float(archive["scaler_std"])
    teacher = teacher_normalized * scaler_std + scaler_mean
    del teacher_normalized
    with np.load(paths["raw"]) as archive:
        raw = archive["data"].astype(np.float32)[..., 0]
    risk_score = None
    if paths["risk"] is not None:
        with np.load(paths["risk"]) as archive:
            risk_truth = archive["validation_truth"].astype(np.float32)[..., 0]
            risk_score = archive["validation_risk_score"].astype(np.float64).reshape(-1)
    if student.shape != truth.shape or teacher.shape != truth.shape:
        raise ValueError("Student, Teacher, and truth shapes are not aligned")
    if risk_score is not None and (
        risk_truth.shape != truth.shape or not np.array_equal(risk_truth, truth)
    ):
        raise ValueError("Existing risk scores are not aligned with current Validation windows")

    valid = np.isfinite(truth) & (truth != 0.0)
    valid_count = valid.sum(axis=2).astype(np.float64)
    student_sum = np.where(valid, np.abs(student - truth), 0.0).sum(axis=2, dtype=np.float64)
    teacher_sum = np.where(valid, np.abs(teacher - truth), 0.0).sum(axis=2, dtype=np.float64)
    del valid, truth, teacher

    prediction_mean_score = student.mean(axis=(1, 2), dtype=np.float64)
    features, volatility_score = build_features(raw, endpoints, student)
    del raw, student

    total = len(endpoints)
    fit_end = total // 2
    selection_end = 3 * total // 4
    fit_index = np.arange(0, fit_end)
    selection_index = np.arange(fit_end + EMBARGO, selection_end)
    evaluation_index = np.arange(selection_end + EMBARGO, total)
    inner_train_end = 3 * fit_end // 4
    inner_train_index = np.arange(0, inner_train_end)
    inner_validation_index = np.arange(inner_train_end + EMBARGO, fit_end)
    feature_mean = features[fit_index].mean(axis=0, dtype=np.float64)
    feature_std = features[fit_index].std(axis=0, dtype=np.float64)
    feature_std[feature_std < 1e-6] = 1.0
    x = ((features - feature_mean) / feature_std).astype(np.float32)
    fit_utility = split_utility(
        student_sum[fit_index], teacher_sum[fit_index], valid_count[fit_index]
    )
    inner_train_utility = split_utility(
        student_sum[inner_train_index],
        teacher_sum[inner_train_index],
        valid_count[inner_train_index],
    )
    inner_validation_utility = split_utility(
        student_sum[inner_validation_index],
        teacher_sum[inner_validation_index],
        valid_count[inner_validation_index],
    )
    target_mean = float(fit_utility.mean())
    target_std = float(fit_utility.std())
    if target_std < 1e-8:
        target_std = 1.0
    y_fit_scaled = ((fit_utility - target_mean) / target_std).astype(np.float32)
    inner_target_mean = float(inner_train_utility.mean())
    inner_target_std = float(inner_train_utility.std())
    if inner_target_std < 1e-8:
        inner_target_std = 1.0
    y_inner_train_scaled = (
        (inner_train_utility - inner_target_mean) / inner_target_std
    ).astype(np.float32)
    y_inner_validation_scaled = (
        (inner_validation_utility - inner_target_mean) / inner_target_std
    ).astype(np.float32)

    models = []
    training = []
    for seed in SEEDS:
        model, record = train_one(
            seed,
            x[inner_train_index],
            y_inner_train_scaled,
            x[inner_validation_index],
            y_inner_validation_scaled,
            x[fit_index],
            y_fit_scaled,
        )
        models.append(model)
        training.append(record)
        print(
            f"seed={seed} best_epoch={record['best_epoch']} "
            f"inner_validation_loss={record['inner_validation_loss']:.6f}",
            flush=True,
        )

    x_evaluation = torch.from_numpy(x[evaluation_index])
    x_selection = torch.from_numpy(x[selection_index])
    predictions = []
    selection_predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            predictions.append(model(x_evaluation).numpy())
            selection_predictions.append(model(x_selection).numpy())
    gate_score = np.mean(predictions, axis=0)
    selection_gate_score = np.mean(selection_predictions, axis=0)
    eval_student = student_sum[evaluation_index]
    eval_teacher = teacher_sum[evaluation_index]
    eval_count = valid_count[evaluation_index]
    selection_utility = split_utility(
        student_sum[selection_index], teacher_sum[selection_index], valid_count[selection_index]
    )
    eval_utility = split_utility(eval_student, eval_teacher, eval_count)
    student_mae = routed_mae(
        eval_student, eval_teacher, eval_count, np.empty(0, dtype=np.int64)
    )
    teacher_mae = routed_mae(
        eval_student, eval_teacher, eval_count, np.arange(len(eval_student))
    )

    methods = {
        "historical_volatility": evaluate_method(
            volatility_score[evaluation_index], eval_student, eval_teacher, eval_count
        ),
        "student_prediction_mean": evaluate_method(
            prediction_mean_score[evaluation_index], eval_student, eval_teacher, eval_count
        ),
        "learned_utility_gate_ensemble": evaluate_method(
            gate_score, eval_student, eval_teacher, eval_count
        ),
        "truth_utility_oracle": evaluate_method(
            eval_utility, eval_student, eval_teacher, eval_count
        ),
    }
    if risk_score is not None:
        methods["existing_student_risk"] = evaluate_method(
            risk_score[evaluation_index], eval_student, eval_teacher, eval_count
        )
    strict_online_methods = {
        "historical_volatility": evaluate_fixed_thresholds(
            volatility_score[selection_index],
            volatility_score[evaluation_index],
            eval_student,
            eval_teacher,
            eval_count,
        ),
        "student_prediction_mean": evaluate_fixed_thresholds(
            prediction_mean_score[selection_index],
            prediction_mean_score[evaluation_index],
            eval_student,
            eval_teacher,
            eval_count,
        ),
        "learned_utility_gate_ensemble": evaluate_fixed_thresholds(
            selection_gate_score,
            gate_score,
            eval_student,
            eval_teacher,
            eval_count,
        ),
    }
    if risk_score is not None:
        strict_online_methods["existing_student_risk"] = evaluate_fixed_thresholds(
            risk_score[selection_index],
            risk_score[evaluation_index],
            eval_student,
            eval_teacher,
            eval_count,
        )
    safe_budget_threshold = evaluate_safe_budget_thresholds(
        selection_gate_score,
        selection_utility,
        gate_score,
        eval_student,
        eval_teacher,
        eval_count,
    )

    random_generator = np.random.default_rng(20261016)
    random_result = {}
    for budget in BUDGETS:
        count = max(1, int(math.floor(budget * len(evaluation_index))))
        maes = []
        for _ in range(500):
            chosen = random_generator.choice(len(evaluation_index), size=count, replace=False)
            maes.append(routed_mae(eval_student, eval_teacher, eval_count, chosen))
        random_result[str(budget)] = {
            "teacher_windows": count,
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "gain_vs_student_mean": student_mae - float(np.mean(maes)),
        }
    methods["random_routing_500_trials"] = random_result

    learned_10 = methods["learned_utility_gate_ensemble"]["0.1"]["gain_vs_student"]
    oracle_10 = methods["truth_utility_oracle"]["0.1"]["gain_vs_student"]
    result = {
        "dataset": dataset,
        "protocol_version": "v2_nested_fit_official_mae_strict_online_budget",
        "validation_only": True,
        "test_keys_read": False,
        "protocol": {
            "total_validation_windows": total,
            "fit": [int(fit_index[0]), int(fit_index[-1])],
            "inner_train": [int(inner_train_index[0]), int(inner_train_index[-1])],
            "inner_validation": [
                int(inner_validation_index[0]),
                int(inner_validation_index[-1]),
            ],
            "selection": [int(selection_index[0]), int(selection_index[-1])],
            "evaluation": [int(evaluation_index[0]), int(evaluation_index[-1])],
            "embargo_windows": EMBARGO,
            "gate_inputs": "past 12 raw observations, frozen Student 12-step forecast, time-of-day/day-of-week",
            "model_selection": "epoch chosen only inside Fit by chronological inner validation loss; refit on all Fit",
            "routing": "exact top-k is an offline diagnostic only",
            "streaming_check": "threshold fixed from Selection; chronological token bucket enforces the hard budget on Evaluation",
            "metric": "official masked horizon-wise MAE averaged over 12 horizons",
        },
        "feature_dimension": int(features.shape[1]),
        "training": training,
        "evaluation": {
            "num_windows": int(len(evaluation_index)),
            "student_mae": student_mae,
            "teacher_always_mae": teacher_mae,
            "teacher_always_gain": student_mae - teacher_mae,
            "teacher_better_window_fraction": float(np.mean(eval_utility > 0.0)),
            "learned_gate_utility_spearman": spearman(gate_score, eval_utility),
            "methods": methods,
            "strict_online_methods": strict_online_methods,
            "safe_up_to_budget_streaming": safe_budget_threshold,
            "learned_oracle_recovery_at_10pct": (
                learned_10 / oracle_10 if oracle_10 > 0 else None
            ),
        },
        "screening_rule": {
            "gain_at_10pct_at_least": 0.03,
        "must_beat_random_and_available_baselines": True,
            "oracle_recovery_at_10pct_at_least": 0.60,
        },
    }
    learned = methods["learned_utility_gate_ensemble"]["0.1"]
    random_10 = methods["random_routing_500_trials"]["0.1"]
    strict_learned = strict_online_methods["learned_utility_gate_ensemble"]["0.1"]
    strict_comparison_maes = [
        random_10["mae_mean"],
        strict_online_methods["historical_volatility"]["0.1"]["mae"],
        strict_online_methods["student_prediction_mean"]["0.1"]["mae"],
    ]
    if "existing_student_risk" in strict_online_methods:
        strict_comparison_maes.append(
            strict_online_methods["existing_student_risk"]["0.1"]["mae"]
        )
    strict_recovery = (
        strict_learned["gain_vs_student"] / oracle_10 if oracle_10 > 0 else None
    )
    result["evaluation"]["strict_online_oracle_recovery_at_10pct"] = strict_recovery
    result["screening_pass"] = bool(
        strict_learned["gain_vs_student"] >= 0.03
        and strict_learned["realized_evaluation_budget"] <= 0.10 + 1e-12
        and all(strict_learned["mae"] < value for value in strict_comparison_maes)
        and strict_recovery >= 0.60
    )
    result["relative_screening_pass"] = bool(
        strict_learned["gain_vs_student"] > 0.0
        and strict_learned["realized_evaluation_budget"] <= 0.10 + 1e-12
        and all(strict_learned["mae"] < value for value in strict_comparison_maes)
        and strict_recovery >= 0.60
    )
    paths["result"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    score_payload = dict(
        evaluation_indices=evaluation_index,
        evaluation_endpoints=endpoints[evaluation_index],
        learned_gate_score=gate_score,
        selection_gate_score=selection_gate_score,
        true_utility=eval_utility,
        historical_volatility_score=volatility_score[evaluation_index],
    )
    if risk_score is not None:
        score_payload["existing_student_risk_score"] = risk_score[evaluation_index]
    np.savez_compressed(paths["scores"], **score_payload)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
