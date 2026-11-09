"""Build a physically sparse sensor cache from a train-memory archive.

Selection uses only the first 55% of Validation. Test retrieval values are
subselected as deployable inputs, but Test labels/predictions are not read.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import zipfile
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--no-memory", required=True, type=Path)
    parser.add_argument("--full-memory", required=True, type=Path)
    parser.add_argument("--selected-nodes", type=int,
                        help="Required for original Top-k mode; optional with --node-ids-json")
    parser.add_argument("--node-ids-json", type=Path,
                        help="Use frozen IDs from a budget plan instead of recomputing Top-k")
    parser.add_argument("--validation-only", action="store_true",
                        help="Package train/val query payload only; never read Test members")
    parser.add_argument("--output-cache", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_cache.exists() or args.output_report.exists():
        raise FileExistsError("Refusing to overwrite sparse-cache artifacts")
    with np.load(args.no_memory, allow_pickle=False) as archive:
        no_memory = np.asarray(archive["val_prediction"], dtype=np.float64)
        truth = np.asarray(archive["val_truth"], dtype=np.float64)
        endpoints = np.asarray(archive["val_endpoints"])
    with np.load(args.full_memory, allow_pickle=False) as archive:
        full_memory = np.asarray(archive["val_prediction"], dtype=np.float64)
        if not np.array_equal(truth, archive["val_truth"]):
            raise ValueError("Validation labels differ")
        if not np.array_equal(endpoints, archive["val_endpoints"]):
            raise ValueError("Validation endpoints differ")
    train_end = int(0.55 * len(truth))
    valid = np.round(truth[:train_end]) > 0
    count = valid.sum(axis=(0, 1))
    gain = (
        (np.abs(no_memory[:train_end] - truth[:train_end]) * valid).sum(axis=(0, 1))
        - (np.abs(full_memory[:train_end] - truth[:train_end]) * valid).sum(axis=(0, 1))
    ) / count
    if args.node_ids_json:
        with args.node_ids_json.open("r", encoding="utf-8") as stream:
            node_ids = np.asarray(json.load(stream)["selected_node_ids"], dtype=np.int32)
        if ((args.selected_nodes is not None and len(node_ids) != args.selected_nodes)
                or len(np.unique(node_ids)) != len(node_ids)):
            raise ValueError("Plan IDs are not unique or do not match --selected-nodes")
        if np.any((node_ids < 0) | (node_ids >= len(gain))):
            raise ValueError("Plan IDs out of range")
        node_ids = np.sort(node_ids)
    else:
        if args.selected_nodes is None or not 0 < args.selected_nodes <= len(gain):
            raise ValueError("Top-k mode requires a valid --selected-nodes")
        node_ids = np.sort(
            np.argsort(-gain, kind="stable")[:args.selected_nodes]
        ).astype(np.int32)
    source = args.source_cache.resolve()
    destination = args.output_cache.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {temporary}")
    omitted_prefixes = ("future_dispersion_", "history_distance_")
    omitted_members = []
    with zipfile.ZipFile(source, "r") as source_zip:
        compressed_sizes = {
            item.filename: item.compress_size for item in source_zip.infolist()
        }
    try:
        with np.load(source, allow_pickle=False) as archive, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=6, allowZip64=True,
        ) as output_zip:
            with output_zip.open("selected_node_ids.npy", "w", force_zip64=True) as stream:
                np.save(stream, node_ids, allow_pickle=False)
            for key in archive.files:
                if args.validation_only and key.endswith("_test"):
                    omitted_members.append(key + ".npy")
                    continue
                if key.startswith(omitted_prefixes):
                    omitted_members.append(key + ".npy")
                    continue
                value = archive[key]
                if key == "retrieval_quantization_scale":
                    if value.ndim != 3 or value.shape[1] != len(gain):
                        raise ValueError(f"Unexpected quantization scale {value.shape}")
                    value = value[:, node_ids, :]
                if key.startswith(("retrieval_", "exact_retrieval_", "exact_available_")) and key.endswith(("_train", "_val", "_test")):
                    if value.ndim != 4 or value.shape[2] != len(gain):
                        raise ValueError(f"Unexpected node tensor {key}: {value.shape}")
                    value = value[:, :, node_ids, :]
                with output_zip.open(key + ".npy", "w", force_zip64=True) as stream:
                    np.save(stream, value, allow_pickle=False)
                del value
                gc.collect()
        os.replace(temporary, destination)
    except BaseException:
        # Preserve a failed temporary archive for forensic inspection rather
        # than deleting a potentially valuable user artifact implicitly.
        raise
    lean_baseline_bytes = source.stat().st_size - sum(
        compressed_sizes[name] for name in omitted_members
    )
    report = {
        "protocol": "validation_prefix_utility_selected_physical_sparse_cache",
        "test_labels_or_predictions_read": False,
        "test_query_members_read": not args.validation_only,
        "selection_from_plan": str(args.node_ids_json.resolve()) if args.node_ids_json else None,
        "selection_prefix_samples": train_end,
        "selected_nodes": len(node_ids),
        "total_nodes": len(gain),
        "source_cache": str(source),
        "source_cache_bytes": source.stat().st_size,
        "unused_confidence_members_omitted": omitted_members,
        "lean_full_node_baseline_bytes_approximately": lean_baseline_bytes,
        "sparse_cache": str(destination),
        "sparse_cache_bytes": destination.stat().st_size,
        "sparse_cache_mib": destination.stat().st_size / 2**20,
        "selected_node_ids": node_ids.tolist(),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        key: report[key] for key in (
            "selected_nodes", "total_nodes", "source_cache_bytes",
            "lean_full_node_baseline_bytes_approximately", "sparse_cache_bytes",
            "sparse_cache_mib",
        )
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
