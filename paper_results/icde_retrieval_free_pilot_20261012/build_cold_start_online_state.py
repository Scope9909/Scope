"""Convert an audited online state into a causal cold-start streaming state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.online_residual_memory import build_cold_start_state, packed_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("PEMS03", "PEMS04", "PEMS07", "PEMS08"),
                        required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_state.exists() or args.output_report.exists():
        raise FileExistsError("Refusing to overwrite a state or report")
    with np.load(args.source_state, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    length = int(np.asarray(source["input_length"]).item())
    horizon = int(np.asarray(source["predict_length"]).item())
    index_path = (ROOT / "datasets" / args.dataset /
                  f"{args.dataset}_{length}to{horizon}_index_scaler.npz")
    with np.load(index_path, allow_pickle=False) as archive:
        first_query_endpoint = int(np.asarray(archive["idx_val"])[0])
    state = build_cold_start_state(source, first_query_endpoint)
    payload = packed_state(state)
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    args.output_state.write_bytes(payload)
    original_candidates = len(source["idx_train"])
    active_candidates = len(state["idx_train"])
    pending_candidates = len(state["pending_idx_train"])
    report = {
        "protocol": "causal_cold_start_streaming_state_v1",
        "dataset": args.dataset,
        "test_evaluated": False,
        "source_state": str(args.source_state.resolve()),
        "source_state_bytes": args.source_state.stat().st_size,
        "state_bytes": len(payload),
        "first_validation_query_endpoint": first_query_endpoint,
        "snapshot_observed_through": int(state["cold_start_endpoint"]),
        "source_raw_observed_through": len(source["train_raw"]) - 1,
        "unobservable_future_frames_removed": (
            len(source["train_raw"]) - 1 - first_query_endpoint),
        "original_candidates": original_candidates,
        "active_candidates_at_cold_start": active_candidates,
        "pending_candidate_metadata_only": pending_candidates,
        "max_active_future_endpoint": int(
            np.max(state["idx_train"]) + horizon),
        "min_pending_future_endpoint": (None if pending_candidates == 0 else int(
            np.min(state["pending_idx_train"]) + horizon)),
        "raw_pending_futures_serialized": False,
        "query_outputs_serialized": False,
        "validation_labels_serialized": False,
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
