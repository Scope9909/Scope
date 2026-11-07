"""Paired bootstrap: frozen teacher versus final Lite-R256 + ValCal.

The teacher caches are stored in z-score space; final archives are stored in
the original traffic units.  This script restores teacher predictions with the
cache scaler, truncates only documented loader padding, and resamples aligned
test endpoints.  No model is trained or refit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent

DATASETS = {
    "PEMS03": (
        "datasets/PEMS03/PEMS03_12to12_parent_predictions_joint3.npz",
        "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz",
    ),
    "PEMS04": (
        "datasets/PEMS04/PEMS04_12to12_parent_predictions_joint3.npz",
        "paper_results/lite_r256_kdfix3_a30_lr1e4_smooth_tod_s1/selected_predictions.npz",
    ),
    "PEMS07": (
        "datasets/PEMS07/PEMS07_12to12_parent_predictions_joint3.npz",
        "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz",
    ),
    "PEMS08": (
        "datasets/PEMS08/PEMS08_12to12_parent_predictions_joint3.npz",
        "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz",
    ),
    "PEMSBAY": (
        "datasets/PEMSBAY/PEMSBAY_12to12_parent_predictions_final_joint_712.npz",
        "paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz",
    ),
}


def horizon_sums(prediction: np.ndarray, truth: np.ndarray):
    valid = np.isfinite(truth) & (truth != 0)
    reduce_axes = tuple(range(2, truth.ndim))
    return (
        (np.abs(prediction - truth) * valid).sum(axis=reduce_axes),
        valid.sum(axis=reduce_axes),
    )


def horizon_mae(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    sums, counts = horizon_sums(prediction, truth)
    return sums.sum(axis=0) / counts.sum(axis=0)


def load_aligned(teacher_path: Path, final_path: Path):
    with np.load(final_path, allow_pickle=False) as archive:
        final_prediction = np.asarray(archive["test_prediction"], dtype=np.float64)
        truth = np.asarray(archive["test_truth"], dtype=np.float64)
        endpoints = np.asarray(archive["test_endpoints"], dtype=np.int64)
    with np.load(teacher_path, allow_pickle=False) as archive:
        teacher_z = np.asarray(archive["teacher_pred_test"], dtype=np.float64)
        mean = np.asarray(archive["scaler_mean"], dtype=np.float64)
        std = np.asarray(archive["scaler_std"], dtype=np.float64)
    if len(teacher_z) < len(truth):
        raise ValueError(f"teacher cache shorter than official test: {teacher_path}")
    padding_rows = int(len(teacher_z) - len(truth))
    teacher_z = teacher_z[: len(truth)]
    if teacher_z.ndim == 4 and teacher_z.shape[-1] == 1:
        teacher_z = teacher_z[..., 0]
    if mean.size == 1:
        teacher_prediction = teacher_z * float(std.reshape(-1)[0]) + float(mean.reshape(-1)[0])
    else:
        teacher_prediction = teacher_z * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    if teacher_prediction.shape != truth.shape or final_prediction.shape != truth.shape:
        raise ValueError(
            f"shape mismatch: teacher={teacher_prediction.shape}, "
            f"final={final_prediction.shape}, truth={truth.shape}"
        )
    if not np.isfinite(teacher_prediction).all() or not np.isfinite(final_prediction).all():
        raise ValueError("non-finite prediction")
    return teacher_prediction, final_prediction, truth, endpoints, padding_rows


def paired_bootstrap(teacher, final, truth, repeats, seed):
    teacher_sum, counts = horizon_sums(teacher, truth)
    final_sum, final_counts = horizon_sums(final, truth)
    if not np.array_equal(counts, final_counts):
        raise AssertionError("paired masks differ")
    n = len(truth)
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 100):
        size = min(100, repeats - start)
        sample = rng.integers(0, n, size=(size, n), endpoint=False)
        boot_counts = counts[sample].sum(axis=1)
        teacher_mae = (teacher_sum[sample].sum(axis=1) / boot_counts).mean(axis=1)
        final_mae = (final_sum[sample].sum(axis=1) / boot_counts).mean(axis=1)
        differences[start : start + size] = teacher_mae - final_mae
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "protocol": "frozen_teacher_vs_lite_r256_valcal_paired_bootstrap_v1",
        "comparison": "teacher_MAE_minus_full_system_MAE; positive favors full system",
        "resampling_unit": "aligned official test endpoint",
        "aggregation": "masked MAE per horizon, arithmetic mean across 12 horizons",
        "bootstrap_repeats": args.repeats,
        "rng_seed": args.seed,
        "final_system_seed": 1,
        "datasets": {},
    }
    rows = []
    for offset, (dataset, (teacher_rel, final_rel)) in enumerate(DATASETS.items()):
        teacher, final, truth, endpoints, padding = load_aligned(
            ROOT / teacher_rel, ROOT / final_rel
        )
        teacher_h = horizon_mae(teacher, truth)
        final_h = horizon_mae(final, truth)
        diffs = paired_bootstrap(
            teacher, final, truth, args.repeats, args.seed + offset
        )
        observed = float(teacher_h.mean() - final_h.mean())
        record = {
            "samples": int(len(truth)),
            "endpoint_min": int(endpoints.min()),
            "endpoint_max": int(endpoints.max()),
            "teacher_padding_rows_discarded": padding,
            "teacher_mae": float(teacher_h.mean()),
            "full_system_mae": float(final_h.mean()),
            "observed_improvement": observed,
            "relative_improvement_percent": float(100 * observed / teacher_h.mean()),
            "ci95_percentile": [
                float(np.quantile(diffs, 0.025)),
                float(np.quantile(diffs, 0.975)),
            ],
            "probability_improvement_le_zero": float(np.mean(diffs <= 0)),
            "teacher_horizon_mae": teacher_h.tolist(),
            "full_system_horizon_mae": final_h.tolist(),
            "improvement_by_horizon": (teacher_h - final_h).tolist(),
            "all_horizons_improve": bool(np.all(teacher_h - final_h > 0)),
        }
        summary["datasets"][dataset] = record
        rows.append({
            "dataset": dataset,
            "samples": record["samples"],
            "teacher_mae": record["teacher_mae"],
            "full_system_mae": record["full_system_mae"],
            "improvement": record["observed_improvement"],
            "relative_improvement_percent": record["relative_improvement_percent"],
            "ci95_low": record["ci95_percentile"][0],
            "ci95_high": record["ci95_percentile"][1],
            "probability_improvement_le_zero": record["probability_improvement_le_zero"],
            "all_horizons_improve": record["all_horizons_improve"],
        })

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "teacher_vs_full_bootstrap.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
