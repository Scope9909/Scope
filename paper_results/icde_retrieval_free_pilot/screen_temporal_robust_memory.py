"""Exploratory time-stability-aware INT8 memory allocation on Validation only.

All sensor utilities come from the first 55% of Validation; per-node costs
come from Train cache payloads. Future folds and Test are never used for
selection. Physical archive bytes must be audited separately after export.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_heterogeneous_memory import FOLDS, cost_by_node, load, loss_table, score, utility


def select(gain: np.ndarray, cost: np.ndarray, budget: int) -> tuple[np.ndarray, int]:
    order = np.lexsort((np.arange(len(gain)), -gain / cost))
    chosen = np.zeros(len(gain), dtype=np.int8)
    spent = 0
    for node in order:
        if gain[node] <= 0:
            break
        node_cost = int(cost[node])
        if spent + node_cost <= budget:
            chosen[node] = 1
            spent += node_cost
    return chosen, spent


def plan(choices: np.ndarray, topk: np.ndarray, fit_end: int, name: str,
         train_budget: int, train_spent: int) -> dict:
    return {
        "protocol": name,
        "test_evaluated": False,
        "cost_definition": "zlib-6 Train query payload and INT8 scale; actual archive audited separately",
        "folds": [{
            "fit_end": fit_end,
            "topk_int8_node_ids": sorted(int(x) for x in topk),
            "heterogeneous_mode_by_node": choices.tolist(),
            "train_payload_budget_bytes": train_budget,
            "selected_train_payload_bytes": train_spent,
        }],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-predictions", type=Path, required=True)
    parser.add_argument("--int8-predictions", type=Path, required=True)
    parser.add_argument("--int8-cache", type=Path, required=True)
    parser.add_argument("--topk-nodes", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--stability-penalty", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.blocks < 2 or args.stability_penalty < 0:
        raise ValueError("Expected at least two blocks and nonnegative penalty")
    no, truth, endpoints = load(args.backbone_predictions)
    yes, truth2, endpoints2 = load(args.int8_predictions)
    if not (no.shape == yes.shape == truth.shape and
            np.array_equal(truth, truth2) and
            np.array_equal(endpoints, endpoints2)):
        raise ValueError("Prediction/truth/endpoints alignment failed")
    n, _, nodes = truth.shape
    if not 0 < args.topk_nodes <= nodes:
        raise ValueError("Invalid node count")
    predictions = [no.astype(np.float64), yes.astype(np.float64)]
    truth = truth.astype(np.float64)
    fit_end = int(n * FOLDS[0][0])
    bounds = np.linspace(0, fit_end, args.blocks + 1, dtype=int)
    block_gain = np.stack([
        utility(loss_table(predictions, truth, int(start), int(end)))[:, 1]
        for start, end in zip(bounds[:-1], bounds[1:])
    ])
    standard_gain = utility(loss_table(predictions, truth, 0, fit_end))[:, 1]
    robust_gain = block_gain.mean(axis=0) - args.stability_penalty * block_gain.std(axis=0)
    costs = cost_by_node(args.int8_cache, nodes)
    if np.any(costs <= 0):
        raise ValueError("Every selected mode must have positive Train payload cost")
    topk = np.argsort(-standard_gain, kind="stable")[:args.topk_nodes]
    budget = int(costs[topk].sum())
    standard, standard_spent = select(standard_gain, costs, budget)
    robust, robust_spent = select(robust_gain, costs, budget)
    topk_choices = np.zeros(nodes, dtype=np.int8)
    topk_choices[topk] = 1
    folds = []
    for fit_fraction, eval_fraction in FOLDS:
        start, end = int(n * fit_fraction), int(n * eval_fraction)
        losses = loss_table(predictions, truth, start, end)
        folds.append({
            "eval_start": start, "eval_end": end,
            "topk_raw_mae": score(losses, topk_choices),
            "plain_ratio_raw_mae": score(losses, standard),
            "robust_raw_mae": score(losses, robust),
        })
    args.output_dir.mkdir(parents=True)
    for name, choices, spent in (("plain_ratio", standard, standard_spent),
                                ("robust", robust, robust_spent)):
        (args.output_dir / f"{name}_plan.json").write_text(
            json.dumps(plan(choices, topk, fit_end, name, budget, spent), indent=2),
            encoding="utf-8")
    report = {
        "protocol": "first55_validation_three_block_stability_int8_screen",
        "test_evaluated": False,
        "blocks": args.blocks,
        "stability_penalty": args.stability_penalty,
        "topk_nodes": args.topk_nodes,
        "train_payload_budget_bytes": budget,
        "mode_counts": {"topk": int(topk_choices.sum()),
                        "plain_ratio": int(standard.sum()),
                        "robust": int(robust.sum())},
        "train_payload_spent_bytes": {"plain_ratio": standard_spent,
                                      "robust": robust_spent},
        "selected_overlap": int(np.logical_and(standard, robust).sum()),
        "folds": folds,
        "mean_robust_minus_plain_ratio": float(np.mean([
            fold["robust_raw_mae"] - fold["plain_ratio_raw_mae"] for fold in folds])),
    }
    (args.output_dir / "screen.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
