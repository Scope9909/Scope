"""Export a sparse or full-node Train-only online memory state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.online_residual_memory import build_train_only_state, packed_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("PEMS03", "PEMS04", "PEMS07", "PEMS08"), required=True)
    parser.add_argument("--full-cache", "--full-int8-cache", dest="full_int8_cache",
                        type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", type=Path)
    group.add_argument("--all-nodes", action="store_true")
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_state.exists() or args.output_report.exists():
        raise FileExistsError("Refusing to overwrite a state or report")
    with np.load(args.full_int8_cache, allow_pickle=False) as archive:
        total_nodes = (int(np.asarray(archive["retrieval_quantization_scale"]).shape[1])
                       if "retrieval_quantization_scale" in archive.files else
                       int(np.asarray(archive["retrieval_train"]).shape[2]))
    if args.all_nodes:
        node_ids = np.arange(total_nodes, dtype=np.int32)
        source = "all_nodes"
    else:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        modes = np.asarray(plan["folds"][0]["heterogeneous_mode_by_node"])
        if len(modes) != total_nodes or np.any((modes != 0) & (modes != 1)):
            raise ValueError("Plan must contain one OFF/INT8 choice per node")
        node_ids = np.flatnonzero(modes == 1).astype(np.int32)
        source = str(args.plan.resolve())
    state = build_train_only_state(args.dataset, args.full_int8_cache, node_ids)
    payload = packed_state(state)
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_state.write_bytes(payload)
    report = {
        "protocol": "train_only_online_memory_state_export",
        "dataset": args.dataset,
        "test_evaluated": False,
        "selection": source,
        "nodes": len(node_ids),
        "total_nodes": total_nodes,
        "state_bytes": len(payload),
        "retrieval_format": str(np.asarray(state["retrieval_format"]).item()),
        "query_outputs_serialized": False,
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
