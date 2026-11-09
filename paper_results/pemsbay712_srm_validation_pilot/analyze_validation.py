"""Validation-only SRM screen for the frozen PEMS-BAY retrieval Student.

Rank nodes on the first 55% of Validation, then report the untouched suffix.
This script never reads Test predictions or labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibrate_dualscale_quantile_residual import metrics


HERE = Path(__file__).resolve().parent
BASE = HERE / "no_memory_validation_predictions.npz"
FULL = ROOT / "paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz"
OUT = HERE / "validation_screen.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    with np.load(BASE, allow_pickle=False) as archive:
        base = np.asarray(archive["val_prediction"], dtype=np.float32)
        truth = np.asarray(archive["val_truth"], dtype=np.float32)
        endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
    with np.load(FULL, allow_pickle=False) as archive:
        full = np.asarray(archive["val_prediction"], dtype=np.float32)
        if not np.array_equal(truth, archive["val_truth"]):
            raise ValueError("Truth mismatch")
        if not np.array_equal(endpoints, archive["val_endpoints"]):
            raise ValueError("Endpoint mismatch")
    if base.shape != full.shape or base.shape != truth.shape:
        raise ValueError("Prediction shape mismatch")
    n, _, nodes = truth.shape
    prefix_end = int(0.55 * n)
    valid = np.rint(truth[:prefix_end]) > 0
    count = valid.sum(axis=(0, 1))
    if np.any(count == 0):
        raise ValueError("Missing valid node labels in selection prefix")
    base_error = (np.abs(base[:prefix_end] - truth[:prefix_end]) * valid).sum(axis=(0, 1))
    full_error = (np.abs(full[:prefix_end] - truth[:prefix_end]) * valid).sum(axis=(0, 1))
    gain = base_error / count - full_error / count
    order = np.argsort(-gain, kind="stable")
    bounds = [prefix_end, int(0.70 * n), int(0.85 * n), n]
    report = {
        "dataset": "PEMSBAY",
        "protocol": "first55_validation_node_gain_screen_v1",
        "checkpoint": "pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1",
        "test_evaluated": False,
        "selection_prefix_end": prefix_end,
        "validation_samples": n,
        "node_count": nodes,
        "positive_gain_nodes": int((gain > 0).sum()),
        "prefix_gain_per_node": gain.tolist(),
        "budgets": [],
    }
    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        keep = round(fraction * nodes)
        selected = np.sort(order[:keep])
        sparse = base.copy()
        sparse[:, :, selected] = full[:, :, selected]
        folds = [metrics(sparse[a:b], truth[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
        report["budgets"].append({
            "fraction": fraction,
            "selected_nodes": selected.tolist(),
            "prefix": metrics(sparse[:prefix_end], truth[:prefix_end]),
            "suffix": metrics(sparse[prefix_end:], truth[prefix_end:]),
            "later_folds": folds,
        })
        print(f"fraction={fraction:.1f} nodes={keep} "
              f"prefix_MAE={report['budgets'][-1]['prefix']['mae']:.6f} "
              f"suffix_MAE={report['budgets'][-1]['suffix']['mae']:.6f}", flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
