"""Validation-only, out-of-sample controls for DCLD calibration.

This deliberately never reads Test arrays. It uses the same three expanding
Validation folds as calibrate_dualscale_smooth_tod.py, with no tuned parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_dualscale_quantile_residual import metrics


FOLDS = ((0.55, 0.70), (0.70, 0.85), (0.85, 1.00))


def fit_bias(prediction: np.ndarray, truth: np.ndarray, mode: str) -> np.ndarray:
    valid = np.round(truth) > 0
    residual = truth.astype(np.float64) - prediction.astype(np.float64)
    if mode == "horizon":
        return np.asarray(
            [np.median(residual[:, h][valid[:, h]]) for h in range(truth.shape[1])]
        )[None, :, None]
    if mode == "node_horizon":
        fallback = fit_bias(prediction, truth, "horizon")[0, :, 0]
        table = np.empty(truth.shape[1:], dtype=np.float64)
        for h in range(truth.shape[1]):
            for n in range(truth.shape[2]):
                mask = valid[:, h, n]
                table[h, n] = (
                    np.median(residual[:, h, n][mask]) if mask.any() else fallback[h]
                )
        return table[None]
    if mode == "global":
        return np.asarray(np.median(residual[valid]))
    raise ValueError(mode)


def fit_affine(prediction: np.ndarray, truth: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary least-squares scale and intercept on valid Validation labels."""
    valid = np.round(truth) > 0
    x = prediction.astype(np.float64)
    y = truth.astype(np.float64)
    if mode == "global":
        x_valid, y_valid = x[valid], y[valid]
        x_mean, y_mean = x_valid.mean(), y_valid.mean()
        variance = np.square(x_valid - x_mean).sum()
        slope = np.dot(x_valid - x_mean, y_valid - y_mean) / variance
        return np.asarray(slope), np.asarray(y_mean - slope * x_mean)
    if mode == "node_horizon":
        count = valid.sum(axis=0)
        safe_count = np.maximum(count, 1)
        x_mean = (x * valid).sum(axis=0) / safe_count
        y_mean = (y * valid).sum(axis=0) / safe_count
        centered_x = (x - x_mean[None]) * valid
        centered_y = (y - y_mean[None]) * valid
        variance = np.square(centered_x).sum(axis=0)
        covariance = (centered_x * centered_y).sum(axis=0)
        slope = np.divide(
            covariance, variance, out=np.ones_like(covariance), where=variance > 1e-12
        )
        intercept = y_mean - slope * x_mean
        intercept[count == 0] = 0.0
        return slope[None], intercept[None]
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    # Load only Validation keys. Test labels and predictions remain unopened.
    with np.load(args.predictions, allow_pickle=False) as archive:
        prediction = np.asarray(archive["val_prediction"], dtype=np.float64)
        truth = np.asarray(archive["val_truth"], dtype=np.float64)
        endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("Validation arrays must have identical [S,H,N] shapes")
    if len(endpoints) != len(prediction) or not np.all(np.diff(endpoints) > 0):
        raise ValueError("Validation endpoints are not strictly aligned")

    folds = []
    for start_fraction, end_fraction in FOLDS:
        fit_end = int(len(prediction) * start_fraction)
        eval_end = int(len(prediction) * end_fraction)
        fit_p, fit_y = prediction[:fit_end], truth[:fit_end]
        eval_p, eval_y = prediction[fit_end:eval_end], truth[fit_end:eval_end]
        scalar = fit_bias(fit_p, fit_y, "global")
        horizon = fit_bias(fit_p, fit_y, "horizon")
        node_horizon = fit_bias(fit_p, fit_y, "node_horizon")
        global_scale, global_intercept = fit_affine(fit_p, fit_y, "global")
        node_scale, node_intercept = fit_affine(fit_p, fit_y, "node_horizon")
        folds.append({
            "fit_samples": fit_end,
            "eval_samples": eval_end - fit_end,
            "network_only": metrics(eval_p, eval_y),
            "global_median_bias": metrics(eval_p + scalar, eval_y),
            "horizon_median_bias": metrics(eval_p + horizon, eval_y),
            "node_horizon_median_bias": metrics(eval_p + node_horizon, eval_y),
            "global_affine": metrics(eval_p * global_scale + global_intercept, eval_y),
            "node_horizon_affine": metrics(eval_p * node_scale + node_intercept, eval_y),
        })

    report = {
        "protocol": "fixed_3_expanding_validation_folds_no_test_access",
        "source_predictions": str(args.predictions.resolve()),
        "bias_estimator": "median residual on valid labels (round(truth)>0)",
        "folds": folds,
        "mean_fold_mae": {
            name: float(np.mean([fold[name]["mae"] for fold in folds]))
            for name in (
                "network_only", "global_median_bias", "horizon_median_bias",
                "node_horizon_median_bias",
                "global_affine", "node_horizon_affine",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["mean_fold_mae"], indent=2))


if __name__ == "__main__":
    main()
