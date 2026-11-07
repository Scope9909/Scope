"""Three-seed Validation-only replication of sensor-utility selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def masked_errors(prediction, truth, subset):
    valid = np.round(truth[subset]) > 0
    error = np.abs(prediction[subset] - truth[subset]) * valid
    return error.sum(axis=0), valid.sum(axis=(0, 2))


def metric_for_nodes(no_error, full_error, count_by_horizon, selected_nodes):
    per_node = no_error.copy()
    per_node[:, selected_nodes] = full_error[:, selected_nodes]
    return float(np.mean(per_node.sum(axis=1) / count_by_horizon))


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    root = ROOT / "paper_results" / "icde_retrieval_free_pilot_20261012"
    results = []
    for seed in (1, 2, 3):
        no_path = root / f"node_budget_s{seed}_no_memory" / "no_memory_validation_predictions.npz"
        full_path = ROOT / "paper_results" / "icde_generality" / (
            f"pems04_joint3_lite_r256_strictalign_s{seed}"
        ) / "selected_predictions.npz"
        with np.load(no_path, allow_pickle=False) as archive:
            no_prediction = np.asarray(archive["val_prediction"], dtype=np.float64)
            truth = np.asarray(archive["val_truth"], dtype=np.float64)
            endpoints = np.asarray(archive["val_endpoints"])
        with np.load(full_path, allow_pickle=False) as archive:
            full_prediction = np.asarray(archive["val_prediction"], dtype=np.float64)
            if not np.array_equal(truth, archive["val_truth"]):
                raise ValueError(f"Seed {seed} truth mismatch")
            if not np.array_equal(endpoints, archive["val_endpoints"]):
                raise ValueError(f"Seed {seed} endpoints mismatch")
        n = len(truth)
        prefix_end = int(0.55 * n)
        prefix_slice = slice(0, prefix_end)
        suffix_slice = slice(prefix_end, n)
        prefix_no, prefix_count = masked_errors(no_prediction, truth, prefix_slice)
        prefix_full, _ = masked_errors(full_prediction, truth, prefix_slice)
        if np.any(prefix_count == 0):
            raise ValueError("A horizon has no valid target in prefix")
        node_count = truth.shape[2]
        per_node_prefix_valid = (np.round(truth[:prefix_end]) > 0).sum(axis=(0, 1))
        gain = (
            (prefix_no - prefix_full).sum(axis=0) / per_node_prefix_valid
        )
        ranking = np.argsort(-gain, kind="stable")
        suffix_no, suffix_count = masked_errors(no_prediction, truth, suffix_slice)
        suffix_full, _ = masked_errors(full_prediction, truth, suffix_slice)
        no_mae = float(np.mean(suffix_no.sum(axis=1) / suffix_count))
        full_mae = float(np.mean(suffix_full.sum(axis=1) / suffix_count))
        rng = np.random.default_rng(20261012 + seed)
        budgets = []
        for fraction in (0.2, 0.4, 0.6, 0.8):
            k = round(fraction * node_count)
            selected_mae = metric_for_nodes(
                suffix_no, suffix_full, suffix_count, ranking[:k]
            )
            random_maes = np.asarray([
                metric_for_nodes(
                    suffix_no, suffix_full, suffix_count,
                    rng.choice(node_count, size=k, replace=False),
                )
                for _ in range(1000)
            ])
            budgets.append({
                "selected_nodes": k,
                "fraction": k / node_count,
                "selected_suffix_mae": selected_mae,
                "selected_minus_full_suffix_mae": selected_mae - full_mae,
                "random_mean_suffix_mae": float(random_maes.mean()),
                "random_fraction_beating_selected": float(np.mean(random_maes <= selected_mae)),
            })
        chronological_folds = []
        for train_fraction, end_fraction in ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0)):
            train_end = int(train_fraction * n)
            eval_end = int(end_fraction * n)
            fit_no, _ = masked_errors(no_prediction, truth, slice(0, train_end))
            fit_full, _ = masked_errors(full_prediction, truth, slice(0, train_end))
            fit_valid_by_node = (np.round(truth[:train_end]) > 0).sum(axis=(0, 1))
            fold_gain = (fit_no - fit_full).sum(axis=0) / fit_valid_by_node
            fold_ranking = np.argsort(-fold_gain, kind="stable")
            eval_no, eval_count = masked_errors(
                no_prediction, truth, slice(train_end, eval_end)
            )
            eval_full, _ = masked_errors(
                full_prediction, truth, slice(train_end, eval_end)
            )
            fold_full_mae = float(np.mean(eval_full.sum(axis=1) / eval_count))
            fold_budgets = []
            for fraction in (0.2, 0.4, 0.6, 0.8):
                k = round(fraction * node_count)
                selected_mae = metric_for_nodes(
                    eval_no, eval_full, eval_count, fold_ranking[:k]
                )
                fold_budgets.append({
                    "selected_nodes": k,
                    "selected_mae": selected_mae,
                    "selected_minus_full_mae": selected_mae - fold_full_mae,
                })
            chronological_folds.append({
                "selection_samples": train_end,
                "evaluation_start": train_end,
                "evaluation_end": eval_end,
                "full_memory_mae": fold_full_mae,
                "budgets": fold_budgets,
            })
        results.append({
            "seed": seed,
            "selection_prefix_samples": prefix_end,
            "evaluation_suffix_samples": n - prefix_end,
            "no_memory_suffix_mae": no_mae,
            "full_memory_suffix_mae": full_mae,
            "budgets": budgets,
            "chronological_folds": chronological_folds,
        })
    summary = []
    for budget_index in range(4):
        values = np.asarray([
            item["budgets"][budget_index]["selected_minus_full_suffix_mae"]
            for item in results
        ])
        summary.append({
            "selected_nodes": results[0]["budgets"][budget_index]["selected_nodes"],
            "selected_minus_full_mae_mean": float(values.mean()),
            "selected_minus_full_mae_sample_std": float(values.std(ddof=1)),
            "individual_differences": values.tolist(),
        })
    report = {
        "protocol": "PEMS04_three_seed_validation_prefix_sensor_utility_selection",
        "test_evaluated": False,
        "sparse_cache_real_forward_and_storage_measured_for_seed_1_only": True,
        "seeds_2_and_3_are_algebraic_sensor_mixes_from_frozen_outputs": True,
        "random_trials_per_seed_budget": 1000,
        "seeds": results,
        "summary": summary,
        "chronological_fold_summary": [
            {
                "selected_nodes": results[0]["budgets"][budget_index]["selected_nodes"],
                "selected_minus_full_mae_mean": float(np.mean([
                    fold["budgets"][budget_index]["selected_minus_full_mae"]
                    for item in results for fold in item["chronological_folds"]
                ])),
                "improved_fold_seed_count": int(np.sum([
                    fold["budgets"][budget_index]["selected_minus_full_mae"] < 0
                    for item in results for fold in item["chronological_folds"]
                ])),
                "total_fold_seed_count": 9,
            }
            for budget_index in range(4)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
