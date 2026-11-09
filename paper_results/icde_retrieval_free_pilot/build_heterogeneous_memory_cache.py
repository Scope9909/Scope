"""Build a Validation-only, physically grouped OFF/INT8/FP16/FP32 cache."""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "fp32_cache", "fp16_cache", "int8_cache",
                 "output_cache", "output_report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--topk-int8-baseline", action="store_true",
                        help="Use first-fold Top-k IDs from the same plan in a grouped INT8 archive")
    args = parser.parse_args()
    if args.output_cache.exists() or args.output_report.exists():
        raise FileExistsError("Refusing to overwrite an existing artifact")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    choices = np.asarray(plan["folds"][0]["heterogeneous_mode_by_node"], dtype=np.int8)
    if len(choices) == 0 or np.any((choices < 0) | (choices > 3)):
        raise ValueError("Invalid mode plan")
    if args.topk_int8_baseline:
        ids = np.asarray(plan["folds"][0]["topk_int8_node_ids"], dtype=np.int64)
        if len(ids) == 0 or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= len(choices))):
            raise ValueError("Invalid Top-k baseline IDs")
        choices[:] = 0
        choices[ids] = 1
    sources = {"int8": args.int8_cache, "fp16": args.fp16_cache,
               "fp32": args.fp32_cache}
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_cache.with_suffix(".tmp.npz")
    if temporary.exists():
        raise FileExistsError(temporary)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as output:
        def write(key, value):
            with output.open(key + ".npy", "w", force_zip64=True) as stream:
                np.save(stream, value, allow_pickle=False)

        write("heterogeneous_memory_format", np.asarray("grouped_v1"))
        with np.load(args.int8_cache, allow_pickle=False) as source:
            for key in ("idx_train", "idx_val", "k", "min_valid_history",
                        "input_length", "predict_length", "scaler_mean", "scaler_std",
                        "train_leave_one_out", "exact_match_rule",
                        "exact_min_valid_history", "exact_train_leave_one_out",
                        "retrieval_quantization", "retrieval_quantization_fit_split"):
                write(key, source[key])
        for code, mode in enumerate(("off", "int8", "fp16", "fp32")):
            if code == 0:
                continue
            ids = np.flatnonzero(choices == code).astype(np.int32)
            write(f"node_ids_{mode}", ids)
            if len(ids) == 0:
                continue
            with np.load(sources[mode], allow_pickle=False) as source:
                for split in ("train", "val"):
                    for prefix in ("retrieval", "exact_retrieval", "exact_available"):
                        key = f"{prefix}_{split}"
                        value = source[key]
                        if value.ndim != 4 or value.shape[2] != len(choices):
                            raise ValueError(f"Unexpected {key} shape in {mode}")
                        write(f"{key}_{mode}", value[:, :, ids, :])
                if mode == "int8":
                    if (str(np.asarray(source["retrieval_quantization_fit_split"]).item())
                            != "train_only"):
                        raise ValueError("INT8 quantization must be train-only")
                    scale = source["retrieval_quantization_scale"]
                    write("retrieval_quantization_scale_int8", scale[:, ids, :])
    os.replace(temporary, args.output_cache)
    report = {
        "protocol": "validation_only_physical_grouped_memory",
        "test_evaluated": False,
        "mode_counts": {mode: int(np.sum(choices == code)) for code, mode in
                        enumerate(("off", "int8", "fp16", "fp32"))},
        "cache_bytes": args.output_cache.stat().st_size,
        "cache_mib": args.output_cache.stat().st_size / 2**20,
        "plan": str(args.plan.resolve()),
        "cache": str(args.output_cache.resolve()),
        "topk_int8_baseline": args.topk_int8_baseline,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
