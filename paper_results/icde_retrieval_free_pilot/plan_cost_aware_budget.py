"""Validation-only seed-1 budget plan. No Test member is accessed.

Cost is zlib-compressed TRAIN query-payload bytes per sensor, not the size of
an online retrieval index. All selection uses past Validation labels only.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np


KEYS = ("retrieval_train", "exact_retrieval_train", "exact_available_train")
FOLDS = ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0))


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--costs-from-report", type=Path,
                   help="Reuse measured per-node costs from an earlier seed with the same cache")
    p.add_argument("--no-memory", type=Path, required=True)
    p.add_argument("--full-memory", type=Path, required=True)
    p.add_argument("--topk-nodes", type=int, default=177)
    p.add_argument("--random-trials", type=int, default=100)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(np.asarray(archive[k]) for k in
                     ("val_prediction", "val_truth", "val_endpoints"))


def measured_costs(path, nodes):
    costs = np.zeros(nodes, dtype=np.int64)
    with np.load(path, allow_pickle=False) as archive:
        for key in KEYS:
            data = archive[key]
            if data.ndim != 4 or data.shape[2] != nodes:
                raise ValueError(f"Unexpected {key} shape {data.shape}")
            for node in range(nodes):
                costs[node] += len(zlib.compress(
                    np.ascontiguousarray(data[:, :, node, :]).tobytes(), level=6))
            del data
    if np.any(costs <= 0):
        raise ValueError("Every node must have positive measured cost")
    return costs


def gain_table(no, full, truth, end):
    valid = np.round(truth[:end]) > 0
    count = valid.sum(axis=(0, 1))
    if np.any(count == 0):
        raise ValueError("A node has no valid labels in selection prefix")
    delta = (np.abs(no[:end] - truth[:end]) -
             np.abs(full[:end] - truth[:end])) * valid
    return delta.sum(axis=(0, 1)) / count


def choose_ratio(gain, costs, budget):
    order = np.lexsort((np.arange(len(gain)), -gain / costs))
    selected, spent = [], 0
    for node in order:
        if gain[node] <= 0:
            break
        if spent + costs[node] <= budget:
            selected.append(int(node))
            spent += int(costs[node])
    return np.asarray(sorted(selected), dtype=np.int32)


def mae(no, full, truth, start, end, selected):
    prediction = no[start:end].copy()
    prediction[:, :, selected] = full[start:end, :, selected]
    target = truth[start:end]
    valid = np.round(target) > 0
    per_horizon = ((np.abs(prediction - target) * valid).sum(axis=(0, 2)) /
                   valid.sum(axis=(0, 2)))
    return float(per_horizon.mean())


def main():
    args = arguments()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    no, truth, endpoints = load_predictions(args.no_memory)
    full, truth_full, endpoints_full = load_predictions(args.full_memory)
    if (no.shape != full.shape or no.shape != truth.shape or
            not np.array_equal(truth, truth_full) or
            not np.array_equal(endpoints, endpoints_full)):
        raise ValueError("Prediction/truth/endpoints alignment failed")
    if not (np.isfinite(no).all() and np.isfinite(full).all() and np.isfinite(truth).all()):
        raise ValueError("Predictions and targets must be finite")
    n, _, nodes = truth.shape
    if not (0 < args.topk_nodes <= nodes):
        raise ValueError("Invalid --topk-nodes")
    if args.costs_from_report:
        prior = json.loads(args.costs_from_report.read_text(encoding="utf-8"))
        if (Path(prior["source_cache"]).resolve() != args.source_cache.resolve() or
                prior["node_count"] != nodes):
            raise ValueError("Reused costs do not match this source cache")
        costs = np.asarray(prior["cost_bytes_per_node"], dtype=np.int64)
    else:
        costs = measured_costs(args.source_cache, nodes)
    rng = np.random.default_rng(20261013)
    folds = []
    plans = {}
    for fold_index, (fit_fraction, eval_fraction) in enumerate(FOLDS):
        fit_end, eval_end = int(fit_fraction * n), int(eval_fraction * n)
        gain = gain_table(no, full, truth, fit_end)
        topk = np.sort(np.argsort(-gain, kind="stable")[:args.topk_nodes]).astype(np.int32)
        budget = int(costs[topk].sum())
        ratio = choose_ratio(gain, costs, budget)
        random_maes = []
        for _ in range(args.random_trials):
            order = rng.permutation(nodes)
            chosen, spent = [], 0
            for node in order:
                if spent + costs[node] <= budget:
                    chosen.append(int(node))
                    spent += int(costs[node])
            random_maes.append(mae(no, full, truth, fit_end, eval_end, chosen))
        topk_mae = mae(no, full, truth, fit_end, eval_end, topk)
        ratio_mae = mae(no, full, truth, fit_end, eval_end, ratio)
        folds.append({
            "fit_end": fit_end, "eval_start": fit_end, "eval_end": eval_end,
            "train_payload_budget_bytes": budget,
            "topk_nodes": len(topk), "topk_train_payload_bytes": int(costs[topk].sum()),
            "cost_aware_nodes": len(ratio),
            "cost_aware_train_payload_bytes": int(costs[ratio].sum()),
            "full_mae": mae(no, full, truth, fit_end, eval_end, np.arange(nodes)),
            "no_memory_mae": mae(no, full, truth, fit_end, eval_end, []),
            "topk_mae": topk_mae, "cost_aware_mae": ratio_mae,
            "cost_aware_minus_topk_mae": ratio_mae - topk_mae,
            "random_same_budget_mean_mae": float(np.mean(random_maes)),
            "random_same_budget_fraction_better_than_cost_aware":
                float(np.mean(np.asarray(random_maes) <= ratio_mae)),
        })
        if fold_index == 0:
            plans = {"topk": topk.tolist(), "cost_aware": ratio.tolist()}
    report = {
        "protocol": "seed1_validation_only_chronological_measured_train_payload_cost",
        "test_labels_predictions_or_query_payload_read": False,
        "cost_definition": "sum of independently zlib-6 compressed per-node TRAIN query arrays; proxy, not online index or complete NPZ bytes",
        "source_cache": str(args.source_cache.resolve()),
        "no_memory": str(args.no_memory.resolve()),
        "full_memory": str(args.full_memory.resolve()),
        "node_count": nodes, "topk_nodes": args.topk_nodes,
        "cost_bytes_per_node": costs.tolist(), "folds": folds,
        "mean_cost_aware_minus_topk_mae": float(np.mean(
            [fold["cost_aware_minus_topk_mae"] for fold in folds])),
    }
    out.mkdir(parents=True)
    (out / "budget_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, ids in plans.items():
        (out / f"{name}_nodes.json").write_text(
            json.dumps({"selected_node_ids": ids, "selection_fold": 0,
                        "train_payload_budget_bytes": folds[0]["train_payload_budget_bytes"]},
                       indent=2), encoding="utf-8")
    print(json.dumps({"folds": folds, "mean_cost_aware_minus_topk_mae":
                      report["mean_cost_aware_minus_topk_mae"]}, indent=2))


if __name__ == "__main__":
    main()
