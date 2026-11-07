"""Wrap frozen INT8 node IDs as a grouped-cache build plan (Validation only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-ids-json", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.selected_ids_json.read_text(encoding="utf-8"))
    ids = np.asarray(source["selected_node_ids"], dtype=np.int64)
    if (args.nodes <= 0 or len(ids) == 0 or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= args.nodes)
            or source.get("selection_fold") != 0):
        raise ValueError("Expected unique in-range IDs frozen on first Validation fold")
    choices = np.zeros(args.nodes, dtype=np.int8)
    choices[ids] = 1
    result = {
        "protocol": "fixed_first_fold_plain_gain_per_byte_int8",
        "test_evaluated": False,
        "folds": [{"topk_int8_node_ids": sorted(int(x) for x in ids),
                   "heterogeneous_mode_by_node": choices.tolist()}],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected_nodes": len(ids), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
