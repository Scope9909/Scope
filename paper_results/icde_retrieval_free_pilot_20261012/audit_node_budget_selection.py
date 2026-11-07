"""Compare Validation-prefix sensor selection with random same-budget controls.

The frozen model's retrieval and exact adapters act independently per sensor
after the shared backbone. Therefore a sensor mask combines its full-memory
and zero-memory outputs sensor-wise; measured sparse-cache runs are checked
against this algebraic prediction before random controls are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", required=True, type=Path)
    parser.add_argument("--full-memory", required=True, type=Path)
    parser.add_argument("--budget-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


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
            raise ValueError("Validation targets differ")
        if not np.array_equal(endpoints, archive["val_endpoints"]):
            raise ValueError("Validation endpoints differ")
    with args.budget_report.open("r", encoding="utf-8") as stream:
        budget_report = json.load(stream)
    prefix_end = budget_report["selection_prefix_end"]
    mask = np.round(truth[prefix_end:]) > 0
    count_by_horizon = mask.sum(axis=(0, 2))
    no_error = (np.abs(no_memory[prefix_end:] - truth[prefix_end:]) * mask).sum(axis=0)
    full_error = (np.abs(full_memory[prefix_end:] - truth[prefix_end:]) * mask).sum(axis=0)
    no_error_by_horizon_node = no_error.sum(axis=0) if no_error.ndim == 3 else no_error
    full_error_by_horizon_node = full_error.sum(axis=0) if full_error.ndim == 3 else full_error
    # Shapes above are [horizon, node]; a selection replaces those node errors.
    if no_error_by_horizon_node.shape != full_error_by_horizon_node.shape:
        raise ValueError("Error table shapes differ")
    num_nodes = no_error_by_horizon_node.shape[1]

    def subset_mae(node_ids):
        selected = np.zeros(num_nodes, dtype=bool)
        selected[node_ids] = True
        total = no_error_by_horizon_node.copy()
        total[:, selected] = full_error_by_horizon_node[:, selected]
        return float(np.mean(total.sum(axis=1) / count_by_horizon))

    rng = np.random.default_rng(20261012)
    items = []
    for item in budget_report["budgets"]:
        k = item["selected_nodes"]
        if k == num_nodes:
            continue
        with np.load(item["cache_path"], allow_pickle=False) as archive:
            selected = np.asarray(archive["selected_node_ids"], dtype=np.int64)
        if len(selected) != k or len(np.unique(selected)) != k:
            raise ValueError("Sparse cache node list invalid")
        predicted_mae = subset_mae(selected)
        measured_mae = item["suffix_metrics"]["mae"]
        discrepancy = predicted_mae - measured_mae
        if abs(discrepancy) > 0.002:
            raise ValueError(
                f"Sparse-cache run and algebraic prediction differ by {discrepancy}"
            )
        random_maes = np.asarray([
            subset_mae(rng.choice(num_nodes, size=k, replace=False))
            for _ in range(1000)
        ])
        items.append({
            "selected_nodes": k,
            "measured_sparse_mae": measured_mae,
            "algebraic_sparse_mae": predicted_mae,
            "algebraic_minus_measured_mae": discrepancy,
            "random_mean_mae": float(random_maes.mean()),
            "random_p05_mae": float(np.quantile(random_maes, 0.05)),
            "random_p95_mae": float(np.quantile(random_maes, 0.95)),
            "random_fraction_beating_selected": float(np.mean(random_maes <= predicted_mae)),
        })
    report = {
        "protocol": "same_budget_random_sensor_control_on_heldout_validation_suffix",
        "test_evaluated": False,
        "random_trials_per_budget": 1000,
        "random_seed": 20261012,
        "algebraic_check_tolerance_mae": 0.002,
        "budgets": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
