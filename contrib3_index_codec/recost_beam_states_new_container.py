"""Re-cost already evaluated Validation beam states in the new ALSC container.

No predictions, labels, or Test data are loaded.  MAE and stability values are
read from frozen Validation records; only actual package bytes are recomputed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import adaptive_lossless_state_codec as codec
from pilot_reversible_predictive_coding import normalize_state

HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"


def mixed_state(exact: dict[str, np.ndarray], nodes) -> dict[str, np.ndarray]:
    state = {key: np.asarray(value).copy() for key, value in exact.items()}
    raw = np.asarray(state.pop("train_raw"), dtype=np.uint16)
    q_nodes = np.asarray(sorted(set(nodes)), dtype=np.int32)
    exact_nodes = np.setdiff1d(np.arange(raw.shape[1], dtype=np.int32), q_nodes)
    tail_len = int(np.asarray(state["input_length"]).item()) if "cold_start_endpoint" in state else 0
    prefix = raw[:-tail_len] if tail_len else raw
    tail = raw[-tail_len:] if tail_len else raw[:0]
    selected = prefix[:, q_nodes]
    top = selected.max(axis=0) if len(q_nodes) else np.empty(0, dtype=np.float32)
    scale = np.where(top > 0, top / 255.0, 1.0).astype(np.float32)
    state["train_raw_mixed_shape"] = np.asarray(prefix.shape, dtype=np.int32)
    state["train_raw_mixed_q_nodes"] = q_nodes
    state["train_raw_mixed_exact_nodes"] = exact_nodes
    state["train_raw_mixed_q_codes"] = np.clip(
        np.rint(selected.astype(np.float32) / scale[None]), 0, 255).astype(np.uint8)
    state["train_raw_mixed_q_scale"] = scale
    state["train_raw_mixed_exact"] = prefix[:, exact_nodes].astype(np.uint16)
    state["train_raw_mixed_tail"] = tail.astype(np.uint16)
    state["quant_scale"] = state["quant_scale"].astype(np.float16)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    all_results = []
    for dataset in ("pems03", "pems04", "pems07", "pems08"):
        frozen_path = ONLINE / f"{dataset}_s1_beam_threebudget.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        exact, _ = normalize_state(ONLINE / f"{dataset}_s1_val_u16.npz")
        exact_blob, _, _ = codec.encode(exact)
        unique = {}
        for row in frozen["states"]:
            if float(row["val_delta"]) <= 0.02:
                key = tuple(sorted(row["nodes"]))
                unique[key] = row
        rows = []
        for number, (nodes, old) in enumerate(sorted(unique.items(), key=lambda item: (len(item[0]), item[0])), 1):
            blob, _, _ = codec.encode(mixed_state(exact, nodes))
            stability = old["stability"]
            row = dict(nodes=list(nodes), val_delta=float(old["val_delta"]),
                       topk_flip_rate=float(stability["topk_set_flip_rate"]),
                       exact_flip_rate=float(stability["exact_hit_flip_rate"]),
                       old_package_bytes=int(old["bytes"]), new_package_bytes=len(blob),
                       smaller_than_new_exact=len(blob) < len(exact_blob),
                       mae_feasible=float(old["val_delta"]) <= 0.02,
                       threebudget_feasible=(float(old["val_delta"]) <= 0.02 and
                                             float(stability["topk_set_flip_rate"]) <= 0.005 and
                                             float(stability["exact_hit_flip_rate"]) <= 0.01 and
                                             len(blob) < len(exact_blob)))
            rows.append(row)
            print(dataset, number, "/", len(unique), list(nodes), len(blob), flush=True)
            partial = dict(scope="Validation frozen-beam state re-costing",
                           test_used=False, labels_loaded=False,
                           search_rerun=False, new_exact_bytes=len(exact_blob), states=rows)
            (args.output / f"{dataset}.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
        mae_only = [r for r in rows if r["mae_feasible"] and r["smaller_than_new_exact"]]
        three = [r for r in rows if r["threebudget_feasible"]]
        result = dict(dataset=dataset.upper(), candidate_pool=frozen["candidate_nodes"],
                      bounded_states_recosted=len(rows), new_exact_bytes=len(exact_blob),
                      best_mae_only=min(mae_only, key=lambda r: (r["new_package_bytes"], r["val_delta"])) if mae_only else None,
                      best_threebudget=min(three, key=lambda r: (r["new_package_bytes"], r["val_delta"])) if three else None,
                      states=rows)
        all_results.append(result)
    report = dict(scope="Validation-only re-costing of frozen bounded-beam states",
                  test_used=False, labels_loaded=False, predictions_rerun=False,
                  caveat="Only states explored by the old bounded beam are compared; this is not exhaustive search.",
                  results=all_results)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in row.items() if k != "states"} for row in all_results], indent=2))


if __name__ == "__main__":
    main()
