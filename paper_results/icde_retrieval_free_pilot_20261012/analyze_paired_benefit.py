"""Validation-only paired diagnostic for retrieval benefit concentration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-free", required=True, type=Path)
    parser.add_argument("--retrieval-enabled", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def positive_gain_share(gain: np.ndarray, fraction: float) -> float:
    positive = np.sort(np.maximum(gain, 0.0))[::-1]
    return float(positive[: max(1, int(len(positive) * fraction))].sum() / positive.sum())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with np.load(args.retrieval_free, allow_pickle=False) as archive:
        no_cache = np.asarray(archive["val_prediction"], dtype=np.float64)
        truth = np.asarray(archive["val_truth"], dtype=np.float64)
        endpoints = np.asarray(archive["val_endpoints"])
    with np.load(args.retrieval_enabled, allow_pickle=False) as archive:
        retrieval = np.asarray(archive["val_prediction"], dtype=np.float64)
        if not np.array_equal(truth, archive["val_truth"]):
            raise ValueError("Validation labels differ")
        if not np.array_equal(endpoints, archive["val_endpoints"]):
            raise ValueError("Validation endpoints differ")
    if no_cache.shape != retrieval.shape or no_cache.shape != truth.shape:
        raise ValueError("Prediction shapes differ")
    mask = np.round(truth) > 0
    error_no_cache = np.abs(no_cache - truth) * mask
    error_retrieval = np.abs(retrieval - truth) * mask
    gain_by_sample = (
        error_no_cache.sum(axis=(1, 2)) / mask.sum(axis=(1, 2))
        - error_retrieval.sum(axis=(1, 2)) / mask.sum(axis=(1, 2))
    )
    gain_by_node = (
        error_no_cache.sum(axis=(0, 1)) / mask.sum(axis=(0, 1))
        - error_retrieval.sum(axis=(0, 1)) / mask.sum(axis=(0, 1))
    )
    report = {
        "scope": "Validation-only, raw pre-calibration predictions, seed 1",
        "retrieval_free_predictions": str(args.retrieval_free.resolve()),
        "retrieval_enabled_predictions": str(args.retrieval_enabled.resolve()),
        "test_evaluated": False,
        "samples": len(gain_by_sample),
        "nodes": len(gain_by_node),
        "sample_mean_mae_gain_from_retrieval": float(gain_by_sample.mean()),
        "retrieval_better_sample_fraction": float(np.mean(gain_by_sample > 0)),
        "top_20_percent_samples_share_of_positive_gain": positive_gain_share(gain_by_sample, 0.2),
        "retrieval_better_node_fraction": float(np.mean(gain_by_node > 0)),
        "top_20_percent_nodes_share_of_positive_gain": positive_gain_share(gain_by_node, 0.2),
        "interpretation_limit": "Two separately trained models; concentration is diagnostic and does not prove a deployable routing or cache-reduction method.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
