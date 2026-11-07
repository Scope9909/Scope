"""Freeze a utility-ranked INT8 Top-k plan from the first 55% of Validation.

The output is compatible with build_heterogeneous_memory_cache.py. It reads
Validation predictions/labels only and never touches Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from screen_heterogeneous_memory import load, loss_table, utility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-predictions", type=Path, required=True)
    parser.add_argument("--int8-predictions", type=Path, required=True)
    parser.add_argument("--topk-nodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    no, truth, endpoints = load(args.backbone_predictions)
    yes, truth2, endpoints2 = load(args.int8_predictions)
    if not (no.shape == yes.shape == truth.shape and
            np.array_equal(truth, truth2) and
            np.array_equal(endpoints, endpoints2)):
        raise ValueError("Prediction/truth/endpoints alignment failed")
    fit_end = int(len(truth) * 0.55)
    values = utility(loss_table([no.astype(np.float64), yes.astype(np.float64)],
                                truth.astype(np.float64), 0, fit_end))[:, 1]
    if not 0 < args.topk_nodes <= len(values):
        raise ValueError("Invalid Top-k")
    ids = np.argsort(-values, kind="stable")[:args.topk_nodes]
    choices = np.zeros(len(values), dtype=np.int8)
    choices[ids] = 1
    result = {
        "protocol": "fixed_first55_validation_utility_topk_int8",
        "test_evaluated": False,
        "topk_nodes": args.topk_nodes,
        "folds": [{
            "fit_end": fit_end,
            "topk_int8_node_ids": sorted(int(x) for x in ids),
            "heterogeneous_mode_by_node": choices.tolist(),
        }],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"topk_nodes": len(ids), "fit_end": fit_end,
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
