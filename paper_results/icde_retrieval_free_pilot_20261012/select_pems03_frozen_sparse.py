"""Apply the already frozen three-block stability rule to PEMS03 Validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_heterogeneous_memory import cost_by_node, load, loss_table, score, utility
from screen_temporal_robust_memory import select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-predictions", type=Path, required=True)
    parser.add_argument("--full-predictions", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    no, truth, endpoints = load(args.backbone_predictions)
    full, truth_full, endpoints_full = load(args.full_predictions)
    if not (no.shape == full.shape == truth.shape and
            np.array_equal(truth, truth_full) and
            np.array_equal(endpoints, endpoints_full)):
        raise ValueError("PEMS03 Validation predictions are not aligned")
    samples, _, nodes = truth.shape
    topk_nodes = round(0.10 * nodes)
    fit_end = int(0.55 * samples)
    bounds = np.linspace(0, fit_end, 4, dtype=int)
    predictions = [no.astype(np.float64), full.astype(np.float64)]
    truth = truth.astype(np.float64)
    block_gain = np.stack([
        utility(loss_table(predictions, truth, int(start), int(end)))[:, 1]
        for start, end in zip(bounds[:-1], bounds[1:])
    ])
    overall_gain = utility(loss_table(predictions, truth, 0, fit_end))[:, 1]
    robust_gain = block_gain.mean(axis=0) - block_gain.std(axis=0)
    costs = cost_by_node(args.full_cache, nodes)
    topk = np.argsort(-overall_gain, kind="stable")[:topk_nodes]
    budget = int(costs[topk].sum())
    chosen, spent = select(robust_gain, costs, budget)
    if not np.any(chosen):
        raise ValueError("Frozen rule selected no PEMS03 nodes")

    folds = []
    for start_frac, end_frac in ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0)):
        start, end = int(start_frac * samples), int(end_frac * samples)
        losses = loss_table(predictions, truth, start, end)
        folds.append({"eval_start": start, "eval_end": end,
                      "sparse_raw_mae": score(losses, chosen),
                      "full_raw_mae": score(losses, np.ones(nodes, dtype=np.int8)),
                      "backbone_raw_mae": score(losses, np.zeros(nodes, dtype=np.int8))})

    plan = {
        "protocol": "frozen_three_block_stability_float32_validation_seed1",
        "test_evaluated": False,
        "source_seed": 1,
        "selection_fit_samples": fit_end,
        "topk_budget_fraction": 0.10,
        "topk_reference_nodes": topk_nodes,
        "stability_penalty": 1.0,
        "blocks": 3,
        "retrieval_format": "float32",
        "cost_definition": "zlib-6 Train retrieval/exact/availability payload per node",
        "train_payload_budget_bytes": budget,
        "selected_train_payload_bytes": spent,
        "folds": [{"fit_end": fit_end,
                   "heterogeneous_mode_by_node": chosen.tolist()}],
    }
    report = {
        "protocol": plan["protocol"], "test_evaluated": False,
        "nodes": nodes, "selected_nodes": int(chosen.sum()),
        "selected_node_ids": np.flatnonzero(chosen).tolist(),
        "validation_folds": folds,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "screen.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
