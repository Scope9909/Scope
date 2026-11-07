"""Validation-only sparse-node budget screening without writing a large cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", required=True, type=Path)
    parser.add_argument("--full-memory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def error_tables(prediction, truth, start, end):
    valid = np.round(truth[start:end]) > 0
    error = np.abs(prediction[start:end] - truth[start:end]) * valid
    return error.sum(axis=0), valid.sum(axis=(0, 2)), valid.sum(axis=(0, 1))


def evaluate(no_error, full_error, count_by_horizon, selected):
    candidate = no_error.copy()
    candidate[:, selected] = full_error[:, selected]
    return float(np.mean(candidate.sum(axis=1) / count_by_horizon))


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with np.load(args.no_memory, allow_pickle=False) as archive:
        no_memory = np.asarray(archive["val_prediction"], dtype=np.float64)
        truth = np.asarray(archive["val_truth"], dtype=np.float64)
        endpoints = np.asarray(archive["val_endpoints"])
    with np.load(args.full_memory, allow_pickle=False) as archive:
        full_memory = np.asarray(archive["val_prediction"], dtype=np.float64)
        if not np.array_equal(truth, archive["val_truth"]):
            raise ValueError("Validation labels differ")
        if not np.array_equal(endpoints, archive["val_endpoints"]):
            raise ValueError("Validation endpoints differ")
    n, _, node_count = truth.shape
    rng = np.random.default_rng(20261012)
    folds = []
    for train_fraction, end_fraction in ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0)):
        train_end, eval_end = int(n * train_fraction), int(n * end_fraction)
        fit_no, _, node_count_fit = error_tables(no_memory, truth, 0, train_end)
        fit_full, _, _ = error_tables(full_memory, truth, 0, train_end)
        gain = (fit_no - fit_full).sum(axis=0) / node_count_fit
        ranking = np.argsort(-gain, kind="stable")
        eval_no, count_by_horizon, _ = error_tables(no_memory, truth, train_end, eval_end)
        eval_full, _, _ = error_tables(full_memory, truth, train_end, eval_end)
        full_mae = float(np.mean(eval_full.sum(axis=1) / count_by_horizon))
        no_mae = float(np.mean(eval_no.sum(axis=1) / count_by_horizon))
        budgets = []
        for fraction in (0.2, 0.4, 0.6, 0.8):
            k = round(fraction * node_count)
            selected_mae = evaluate(
                eval_no, eval_full, count_by_horizon, ranking[:k]
            )
            random_maes = np.asarray([
                evaluate(
                    eval_no, eval_full, count_by_horizon,
                    rng.choice(node_count, size=k, replace=False),
                )
                for _ in range(1000)
            ])
            budgets.append({
                "selected_nodes": k,
                "selected_mae": selected_mae,
                "selected_minus_full_mae": selected_mae - full_mae,
                "random_mean_mae": float(random_maes.mean()),
                "random_fraction_beating_selected": float(np.mean(random_maes <= selected_mae)),
            })
        folds.append({
            "fit_samples": train_end,
            "eval_start": train_end,
            "eval_end": eval_end,
            "no_memory_mae": no_mae,
            "full_memory_mae": full_mae,
            "budgets": budgets,
        })
    report = {
        "protocol": "Validation-only chronological sensor-utility budget screening",
        "test_evaluated": False,
        "prediction_sources": {
            "no_memory": str(args.no_memory.resolve()),
            "full_memory": str(args.full_memory.resolve()),
        },
        "note": "Algebraic sensor-wise mix; physical sparse cache and runtime forward not measured for this dataset.",
        "num_nodes": node_count,
        "folds": folds,
        "summary": [
            {
                "selected_nodes": folds[0]["budgets"][i]["selected_nodes"],
                "mean_selected_minus_full_mae": float(np.mean([
                    fold["budgets"][i]["selected_minus_full_mae"] for fold in folds
                ])),
                "improved_fold_count": int(np.sum([
                    fold["budgets"][i]["selected_minus_full_mae"] < 0 for fold in folds
                ])),
                "random_better_in_any_fold": bool(np.any([
                    fold["budgets"][i]["random_fraction_beating_selected"] > 0
                    for fold in folds
                ])),
            }
            for i in range(4)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
