"""Validation-only INT8 node allocation with a hard *physical archive* byte cap.

This exploratory allocator ranks nodes by first-55%-Validation MAE gain per
Train payload byte, but admits a node only after serializing the *whole*
grouped Train+Validation archive in memory. It thus repairs the proxy-budget
gap without reading Test or changing the frozen Student. This is an offline
query-cache prototype, not a complete online retrieval index or deployment
storage guarantee.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_heterogeneous_memory import FOLDS, cost_by_node, load, loss_table, score, utility


META = ("idx_train", "idx_val", "k", "min_valid_history", "input_length",
        "predict_length", "scaler_mean", "scaler_std", "train_leave_one_out",
        "exact_match_rule", "exact_min_valid_history", "exact_train_leave_one_out",
        "retrieval_quantization", "retrieval_quantization_fit_split")
PAYLOAD = ("retrieval_train", "exact_retrieval_train", "exact_available_train",
           "retrieval_val", "exact_retrieval_val", "exact_available_val")


def packed_archive_bytes(source: dict[str, np.ndarray], node_ids: np.ndarray) -> int:
    """Mirror build_heterogeneous_memory_cache.py exactly, without disk writes."""
    ids = np.sort(np.asarray(node_ids, dtype=np.int32))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        def write(key: str, value: np.ndarray) -> None:
            with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                np.save(stream, value, allow_pickle=False)

        write("heterogeneous_memory_format", np.asarray("grouped_v1"))
        for key in META:
            write(key, source[key])
        write("node_ids_int8", ids)
        if len(ids):
            for split in ("train", "val"):
                for prefix in ("retrieval", "exact_retrieval", "exact_available"):
                    key = f"{prefix}_{split}"
                    write(f"{key}_int8", source[key][:, :, ids, :])
            write("retrieval_quantization_scale_int8",
                  source["retrieval_quantization_scale"][:, ids, :])
        write("node_ids_fp16", np.empty(0, dtype=np.int32))
        write("node_ids_fp32", np.empty(0, dtype=np.int32))
    return buffer.tell()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-predictions", type=Path, required=True)
    parser.add_argument("--int8-predictions", type=Path, required=True)
    parser.add_argument("--int8-cache", type=Path, required=True)
    parser.add_argument("--byte-budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.byte_budget <= 0:
        raise ValueError("byte budget must be positive")

    no, truth, endpoints = load(args.backbone_predictions)
    full, truth2, endpoints2 = load(args.int8_predictions)
    if not (no.shape == full.shape == truth.shape and
            np.array_equal(truth, truth2) and
            np.array_equal(endpoints, endpoints2)):
        raise ValueError("Prediction/truth/endpoints alignment failed")
    n, _, nodes = truth.shape
    predictions = [no.astype(np.float64), full.astype(np.float64)]
    truth = truth.astype(np.float64)
    fit_end = int(n * FOLDS[0][0])
    gains = utility(loss_table(predictions, truth, 0, fit_end))[:, 1]
    proxy_costs = cost_by_node(args.int8_cache, nodes)
    if np.any(proxy_costs <= 0):
        raise ValueError("Every node requires positive Train payload bytes")
    order = np.lexsort((np.arange(nodes), -gains / proxy_costs))

    with np.load(args.int8_cache, allow_pickle=False) as archive:
        source = {key: archive[key] for key in (*META, *PAYLOAD,
                                              "retrieval_quantization_scale")}
    if (len(source["idx_val"]) != n or
            any(source[key].ndim != 4 or source[key].shape[2] != nodes or
                source[key].shape[0] != (n if key.endswith("_val") else len(source["idx_train"]))
                for key in PAYLOAD)):
        raise ValueError("Cache query counts or node dimensions differ from predictions")
    if str(np.asarray(source["retrieval_quantization_fit_split"]).item()) != "train_only":
        raise ValueError("INT8 scale must be fitted on Train only")
    empty_bytes = packed_archive_bytes(source, np.empty(0, dtype=np.int32))
    if args.byte_budget <= empty_bytes:
        raise ValueError("Byte budget does not exceed archive metadata cost")
    selected: list[int] = []
    actual_bytes = empty_bytes
    attempted = 0
    for node in order:
        node = int(node)
        if gains[node] <= 0:
            break
        attempted += 1
        candidate = np.asarray([*selected, node], dtype=np.int32)
        candidate_bytes = packed_archive_bytes(source, candidate)
        if candidate_bytes <= args.byte_budget:
            selected.append(node)
            actual_bytes = candidate_bytes
    if not selected:
        raise ValueError("No positive-utility node fits the physical byte cap")
    choices = np.zeros(nodes, dtype=np.int8)
    choices[selected] = 1
    proxy_spent = int(proxy_costs[selected].sum())
    deployment_plan = {
        "protocol": "actual_archive_byte_capped_int8",
        "test_evaluated": False,
        "cost_definition": "exact grouped Train+Val ZIP archive bytes; Train-only per-node payload bytes used for ranking",
        "physical_archive_byte_budget": args.byte_budget,
        "physical_archive_bytes_in_memory": actual_bytes,
        "physical_budget_guarantee_scope": "offline Train+Val grouped query archive only",
        "folds": [{
            "fit_end": fit_end,
            "topk_int8_node_ids": sorted(selected),
            "heterogeneous_mode_by_node": choices.tolist(),
            "selected_train_payload_proxy_bytes": proxy_spent,
        }],
    }
    folds = []
    for start_fraction, end_fraction in FOLDS:
        start, end = int(n * start_fraction), int(n * end_fraction)
        losses = loss_table(predictions, truth, start, end)
        folds.append({"eval_start": start, "eval_end": end,
                      "selected_raw_mae": score(losses, choices)})
    report = {
        "protocol": "actual_grouped_archive_byte_capped_int8_validation_screen",
        "test_evaluated": False,
        "selection_prefix_samples": fit_end,
        "byte_budget": args.byte_budget,
        "empty_archive_bytes": empty_bytes,
        "selected_archive_bytes_in_memory": actual_bytes,
        "budget_satisfied": actual_bytes <= args.byte_budget,
        "selected_nodes": len(selected),
        "candidate_nodes_examined": attempted,
        "selected_train_payload_proxy_bytes": proxy_spent,
        "folds": folds,
        "mean_raw_mae": float(np.mean([fold["selected_raw_mae"] for fold in folds])),
        "note": "Byte cap excludes checkpoint, calibrator, candidate-history and online index",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "plan.json").write_text(json.dumps(deployment_plan, indent=2),
                                                encoding="utf-8")
    (args.output_dir / "screen.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
