"""Predefined multi-budget Validation screen against Top-k+INT8.

Computes per-node Train payload costs once and evaluates both a first-fold
frozen deployment plan and the diagnostic fold-wise reselected plan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_heterogeneous_memory import (FOLDS, MODES, allocate, cost_by_node,
                                         load, loss_table, score, utility)


def main():
    p = argparse.ArgumentParser()
    for name in ("backbone_predictions", "int8_predictions", "fp16_predictions",
                 "fp32_predictions", "int8_cache", "fp16_cache", "fp32_cache",
                 "output_dir"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--topk-nodes", nargs="+", type=int, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    ks = sorted(set(args.topk_nodes))
    paths = (args.backbone_predictions, args.int8_predictions,
             args.fp16_predictions, args.fp32_predictions)
    loaded = [load(path) for path in paths]
    truth, endpoints = loaded[0][1:]
    for path, (pred, target, end) in zip(paths, loaded):
        if (pred.shape != truth.shape or not np.array_equal(target, truth) or
                not np.array_equal(end, endpoints)):
            raise ValueError(f"Unaligned predictions: {path}")
    predictions = [item[0].astype(np.float64) for item in loaded]
    truth = truth.astype(np.float64)
    samples, _, nodes = truth.shape
    if any(not 0 < k <= nodes for k in ks):
        raise ValueError("Invalid Top-k node count")
    costs = np.zeros((nodes, len(MODES)), dtype=np.int64)
    for mode, path in ((1, args.int8_cache), (2, args.fp16_cache),
                       (3, args.fp32_cache)):
        costs[:, mode] = cost_by_node(path, nodes)
    precomputed = []
    for fit_fraction, eval_fraction in FOLDS:
        fit_end, eval_end = int(samples * fit_fraction), int(samples * eval_fraction)
        precomputed.append((fit_end, eval_end,
                            utility(loss_table(predictions, truth, 0, fit_end)),
                            loss_table(predictions, truth, fit_end, eval_end)))
    args.output_dir.mkdir(parents=True)
    rows = []
    for k in ks:
        folds = []
        frozen_baseline = frozen_chosen = None
        for index, (fit_end, eval_end, fit_utility, eval_loss) in enumerate(precomputed):
            topk = np.argsort(-fit_utility[:, 1], kind="stable")[:k]
            baseline = np.zeros(nodes, dtype=np.int8)
            baseline[topk] = 1
            budget = int(costs[topk, 1].sum())
            chosen, spent = allocate(fit_utility, costs, budget)
            if index == 0:
                frozen_baseline, frozen_chosen = baseline.copy(), chosen.copy()
            reselected_delta = score(eval_loss, chosen) - score(eval_loss, baseline)
            fixed_topk = score(eval_loss, frozen_baseline)
            fixed_budget = score(eval_loss, frozen_chosen)
            folds.append({
                "fit_end": fit_end, "eval_start": fit_end, "eval_end": eval_end,
                "train_payload_budget_bytes": budget,
                "heterogeneous_train_payload_bytes": spent,
                "topk_int8_mae": fixed_topk,
                "heterogeneous_mae": fixed_budget,
                "heterogeneous_minus_topk_int8_mae": fixed_budget - fixed_topk,
                "reselected_heterogeneous_minus_topk_int8_mae": reselected_delta,
                "mode_counts": {name: int(np.sum(chosen == mode))
                                for mode, name in enumerate(MODES)},
                "topk_int8_node_ids": sorted(int(x) for x in topk) if index == 0 else None,
                "heterogeneous_mode_by_node": chosen.tolist() if index == 0 else None,
            })
        mean_fixed = float(np.mean([f["heterogeneous_minus_topk_int8_mae"]
                                    for f in folds]))
        row = {"topk_nodes": k, "budget_fraction_nodes": k / nodes,
               "first_fold_mode_counts": folds[0]["mode_counts"],
               "mean_fixed_delta": mean_fixed,
               "fixed_folds_won": sum(f["heterogeneous_minus_topk_int8_mae"] < 0
                                      for f in folds),
               "fixed_fold_deltas": [f["heterogeneous_minus_topk_int8_mae"]
                                     for f in folds],
               "mean_reselected_delta": float(np.mean([
                   f["reselected_heterogeneous_minus_topk_int8_mae"]
                   for f in folds]))}
        rows.append(row)
        folder = args.output_dir / f"k{k}"
        folder.mkdir()
        (folder / "screen.json").write_text(json.dumps({
            "protocol": "frozen_first_fold_multi_budget_validation_screen",
            "test_evaluated": False, "modes": MODES, "topk_nodes": k,
            "cost_definition": "independent zlib-6 TRAIN query payload per node/mode",
            "folds": folds, "mean_fixed_delta": mean_fixed,
        }, indent=2), encoding="utf-8")
    summary = {"protocol": "predefined_budget_grid_validation_only",
               "test_evaluated": False, "topk_grid": ks,
               "selection": "first 55 percent of Validation; first-fold plan fixed for all evaluation folds",
               "rows": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
