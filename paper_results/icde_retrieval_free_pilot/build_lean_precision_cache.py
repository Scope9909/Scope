"""Create Validation-only FP16 or train-fitted INT8 retrieval cache.

Only retrieval/exact/availability Train and Validation query payloads plus
audit metadata are packaged. Test and unused confidence members are not read.
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np


META = ("idx_train", "idx_val", "k", "min_valid_history", "input_length",
        "predict_length", "scaler_mean", "scaler_std", "scaler_mode",
        "train_leave_one_out", "calendar_mode", "aggregation",
        "exact_match_rule", "exact_min_valid_history",
        "exact_train_leave_one_out")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--mode", choices=("fp16", "int8"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite cache or report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp.npz")
    if temp.exists():
        raise FileExistsError(temp)

    with np.load(args.source, allow_pickle=False) as source, zipfile.ZipFile(
            temp, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=6, allowZip64=True) as archive:
        def write(key, value):
            with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                np.save(stream, value, allow_pickle=False)

        for key in META:
            if key in source.files:
                write(key, source[key])
        scale = None
        if args.mode == "int8":
            train = np.asarray(source["retrieval_train"], dtype=np.float32)
            scale = np.maximum(np.max(np.abs(train), axis=0) / 127.0,
                               np.finfo(np.float32).eps).astype(np.float32)
            write("retrieval_quantization", np.asarray("symmetric_int8_per_horizon_node"))
            write("retrieval_quantization_fit_split", np.asarray("train_only"))
            write("retrieval_quantization_scale", scale)
        for split in ("train", "val"):
            retrieval = train if split == "train" and args.mode == "int8" else np.asarray(
                source[f"retrieval_{split}"], dtype=np.float32)
            if args.mode == "int8":
                value = np.clip(np.rint(retrieval / scale[None, ...]), -127, 127).astype(np.int8)
            else:
                value = retrieval.astype(np.float16)
            write(f"retrieval_{split}", value)
            del retrieval, value
            exact = source[f"exact_retrieval_{split}"]
            write(f"exact_retrieval_{split}", exact.astype(np.float16))
            del exact
            write(f"exact_available_{split}", source[f"exact_available_{split}"])
    os.replace(temp, args.output)
    with np.load(args.output, allow_pickle=False) as saved:
        if any("test" in name.lower() for name in saved.files):
            raise AssertionError("Test member appeared in Validation-only cache")
        if args.mode == "int8" and not np.array_equal(
                saved["retrieval_quantization_scale"], scale):
            raise AssertionError("INT8 train scale changed during serialization")
    report = {
        "protocol": "lean_validation_only_precision_cache",
        "mode": args.mode, "test_members_read_or_written": False,
        "unused_confidence_members_read_or_written": False,
        "quantization_fit_split": "train_only" if args.mode == "int8" else None,
        "source": str(args.source.resolve()),
        "cache": str(args.output.resolve()),
        "cache_bytes": args.output.stat().st_size,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
