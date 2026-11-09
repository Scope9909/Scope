"""Build PEMS03 Train-only INT8 payload for the frozen Validation protocol.

This deliberately reads no Validation or Test retrieval/target arrays. The
Validation predictions are generated later by causal online replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite an existing result")

    metadata = (
        "idx_train", "scaler_mean", "scaler_std", "k",
        "min_valid_history", "input_length", "predict_length",
        "exact_min_valid_history", "exact_match_rule", "calendar_mode",
        "aggregation",
    )
    with np.load(args.source, allow_pickle=False) as source:
        train = np.asarray(source["retrieval_train"], dtype=np.float32)
        scale = np.maximum(
            np.max(np.abs(train), axis=0) / 127.0,
            np.finfo(np.float32).eps,
        ).astype(np.float32)
        quantized = np.clip(np.rint(train / scale[None, ...]), -127, 127).astype(np.int8)
        del train
        output = {key: np.asarray(source[key]) for key in metadata}
        output["retrieval_train"] = quantized
        output["exact_retrieval_train"] = np.asarray(
            source["exact_retrieval_train"], dtype=np.float16
        )
        output["exact_available_train"] = np.asarray(source["exact_available_train"])
        output["retrieval_quantization"] = np.asarray("symmetric_int8_per_horizon_node")
        output["retrieval_quantization_fit_split"] = np.asarray("train_only")
        output["retrieval_quantization_scale"] = scale

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    report = {
        "protocol": "pems03_train_only_int8_payload_v1",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "fit_split": "train_only",
        "read_validation_retrieval": False,
        "read_test": False,
        "test_evaluated": False,
        "train_queries": int(output["retrieval_train"].shape[0]),
        "nodes": int(output["retrieval_train"].shape[2]),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
