"""Validation-only smooth level and target-time calibration for MiniST.

This is a single-model post-calibrator with two frozen components:

1. Per horizon/node, median residuals are estimated in prediction-quantile
   bins and linearly interpolated as a smooth function of the prediction.
2. Remaining residuals are summarized per node and *forecast target time*
   bucket.  Horizon ``h`` therefore uses ``endpoint + h + 1`` rather than
   incorrectly sharing the input endpoint's time bucket across all horizons.

The default hyper-parameters were selected using Validation-only sweeps and
three expanding chronological folds.  This script repeats the three-fold
Go/No-Go audit and refuses to open Test unless both calibration stages improve
MAE on every fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_dualscale_quantile_residual import metrics


def fit_smooth_level(
    prediction: np.ndarray,
    truth: np.ndarray,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and truth must be equal 3-D arrays")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    _, num_horizons, num_nodes = prediction.shape
    centers = np.empty((num_bins, num_horizons, num_nodes), dtype=np.float64)
    residuals = np.empty_like(centers)
    edge_quantiles = np.arange(1, num_bins, dtype=np.float64) / num_bins
    center_quantiles = (
        np.arange(num_bins, dtype=np.float64) + 0.5
    ) / num_bins

    for horizon in range(num_horizons):
        for node in range(num_nodes):
            valid = np.round(truth[:, horizon, node]) > 0
            if not np.any(valid):
                centers[:, horizon, node] = 0.0
                residuals[:, horizon, node] = 0.0
                continue
            pred = prediction[valid, horizon, node].astype(
                np.float64, copy=False
            )
            error = (
                truth[valid, horizon, node].astype(np.float64, copy=False)
                - pred
            )
            edges = np.quantile(pred, edge_quantiles)
            bin_ids = np.searchsorted(edges, pred, side="right")
            fallback = float(np.median(error))
            for bin_id in range(num_bins):
                selected = bin_ids == bin_id
                if np.any(selected):
                    centers[bin_id, horizon, node] = float(
                        np.median(pred[selected])
                    )
                    residuals[bin_id, horizon, node] = float(
                        np.median(error[selected])
                    )
                else:
                    centers[bin_id, horizon, node] = float(
                        np.quantile(pred, center_quantiles[bin_id])
                    )
                    residuals[bin_id, horizon, node] = fallback
    if not np.isfinite(centers).all() or not np.isfinite(residuals).all():
        raise FloatingPointError("smooth level table contains non-finite values")
    return centers, residuals


def smooth_correction_for(
    prediction: np.ndarray,
    centers: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    if centers.shape != residuals.shape or centers.ndim != 3:
        raise ValueError("centers and residuals must be equal 3-D tables")
    _, num_horizons, num_nodes = centers.shape
    if prediction.shape[1:] != (num_horizons, num_nodes):
        raise ValueError("prediction does not match smooth table dimensions")
    correction = np.empty_like(prediction, dtype=np.float64)
    for horizon in range(num_horizons):
        for node in range(num_nodes):
            x = centers[:, horizon, node]
            values = residuals[:, horizon, node]
            order = np.argsort(x)
            x = x[order]
            values = values[order]
            unique_x, inverse = np.unique(x, return_inverse=True)
            if len(unique_x) != len(x):
                values = np.asarray(
                    [
                        np.median(values[inverse == index])
                        for index in range(len(unique_x))
                    ],
                    dtype=np.float64,
                )
                x = unique_x
            correction[:, horizon, node] = np.interp(
                prediction[:, horizon, node],
                x,
                values,
                left=values[0],
                right=values[-1],
            )
    if not np.isfinite(correction).all():
        raise FloatingPointError("smooth correction contains non-finite values")
    return correction


def target_time_bucket_ids(
    endpoints: np.ndarray,
    num_horizons: int,
    slices_per_day: int,
    num_time_bins: int,
) -> np.ndarray:
    if slices_per_day % num_time_bins != 0:
        raise ValueError("num_time_bins must divide slices_per_day exactly")
    target_offsets = 1 + np.arange(num_horizons, dtype=np.int64)
    target_slots = (
        endpoints.astype(np.int64)[:, None] + target_offsets[None, :]
    ) % slices_per_day
    return target_slots * num_time_bins // slices_per_day


def fit_target_time_residual(
    prediction: np.ndarray,
    truth: np.ndarray,
    endpoints: np.ndarray,
    slices_per_day: int,
    num_time_bins: int,
) -> np.ndarray:
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("time calibration arrays must be equal 3-D arrays")
    if len(endpoints) != len(prediction):
        raise ValueError("endpoints are not aligned with calibration arrays")
    _, num_horizons, num_nodes = prediction.shape
    bucket_ids = target_time_bucket_ids(
        endpoints, num_horizons, slices_per_day, num_time_bins
    )
    table = np.empty((num_time_bins, num_nodes), dtype=np.float64)
    for time_bin in range(num_time_bins):
        in_bin = bucket_ids == time_bin
        for node in range(num_nodes):
            valid = in_bin & (np.round(truth[:, :, node]) > 0)
            selected = (
                truth[:, :, node].astype(np.float64, copy=False)
                - prediction[:, :, node].astype(np.float64, copy=False)
            )[valid]
            table[time_bin, node] = (
                float(np.median(selected)) if selected.size else 0.0
            )
    if not np.isfinite(table).all():
        raise FloatingPointError("target-time residual table is non-finite")
    return table


def target_time_correction_for(
    endpoints: np.ndarray,
    num_horizons: int,
    table: np.ndarray,
    slices_per_day: int,
) -> np.ndarray:
    num_time_bins, num_nodes = table.shape
    bucket_ids = target_time_bucket_ids(
        endpoints, num_horizons, slices_per_day, num_time_bins
    )
    correction = np.empty(
        (len(endpoints), num_horizons, num_nodes), dtype=np.float64
    )
    for horizon in range(num_horizons):
        correction[:, horizon] = table[bucket_ids[:, horizon]]
    return correction


def fit_and_apply(
    fit_prediction: np.ndarray,
    fit_truth: np.ndarray,
    fit_endpoints: np.ndarray,
    query_prediction: np.ndarray,
    query_endpoints: np.ndarray,
    *,
    num_bins: int,
    smooth_shrinkage: float,
    num_time_bins: int,
    time_shrinkage: float,
    slices_per_day: int,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    centers, smooth_residuals = fit_smooth_level(
        fit_prediction, fit_truth, num_bins
    )
    fit_smooth = fit_prediction + smooth_shrinkage * smooth_correction_for(
        fit_prediction, centers, smooth_residuals
    )
    query_smooth = query_prediction + smooth_shrinkage * smooth_correction_for(
        query_prediction, centers, smooth_residuals
    )
    time_table = fit_target_time_residual(
        fit_smooth,
        fit_truth,
        fit_endpoints,
        slices_per_day,
        num_time_bins,
    )
    query_final = query_smooth + time_shrinkage * target_time_correction_for(
        query_endpoints,
        query_prediction.shape[1],
        time_table,
        slices_per_day,
    )
    return query_smooth, query_final, (centers, smooth_residuals, time_table)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            root
            / "paper_results/minist_dualscale_96_720_s1/selected_predictions.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(root / "paper_results/minist_dualscale_smooth_tod_s1"),
    )
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--smooth-shrinkage", type=float, default=0.55)
    parser.add_argument("--num-time-bins", type=int, default=96)
    parser.add_argument("--time-shrinkage", type=float, default=0.35)
    parser.add_argument("--slices-per-day", type=int, default=288)
    args = parser.parse_args()

    if not 0.0 < args.smooth_shrinkage <= 1.0:
        raise ValueError("smooth_shrinkage must be in (0, 1]")
    if not 0.0 < args.time_shrinkage <= 1.0:
        raise ValueError("time_shrinkage must be in (0, 1]")

    # Test keys are not read while the fixed protocol is audited on Validation.
    with np.load(args.predictions, allow_pickle=False) as archive:
        val_prediction = np.asarray(archive["val_prediction"], dtype=np.float64)
        val_truth = np.asarray(archive["val_truth"], dtype=np.float64)
        val_endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
    if val_prediction.shape != val_truth.shape:
        raise ValueError("Validation prediction/truth shapes differ")
    if len(val_endpoints) != len(val_prediction):
        raise ValueError("Validation endpoints are not aligned")
    if not np.all(np.diff(val_endpoints) > 0):
        raise ValueError("Validation endpoints must be strictly increasing")

    fold_fractions = ((0.55, 0.70), (0.70, 0.85), (0.85, 1.00))
    fold_report = []
    for train_fraction, end_fraction in fold_fractions:
        train_end = int(len(val_prediction) * train_fraction)
        eval_end = int(len(val_prediction) * end_fraction)
        smooth, final, _ = fit_and_apply(
            val_prediction[:train_end],
            val_truth[:train_end],
            val_endpoints[:train_end],
            val_prediction[train_end:eval_end],
            val_endpoints[train_end:eval_end],
            num_bins=args.num_bins,
            smooth_shrinkage=args.smooth_shrinkage,
            num_time_bins=args.num_time_bins,
            time_shrinkage=args.time_shrinkage,
            slices_per_day=args.slices_per_day,
        )
        baseline_metrics = metrics(
            val_prediction[train_end:eval_end],
            val_truth[train_end:eval_end],
        )
        smooth_metrics = metrics(
            smooth, val_truth[train_end:eval_end]
        )
        final_metrics = metrics(final, val_truth[train_end:eval_end])
        fold_report.append(
            {
                "train_samples": train_end,
                "evaluation_start": train_end,
                "evaluation_end": eval_end,
                "baseline": baseline_metrics,
                "smooth_level": smooth_metrics,
                "smooth_plus_target_time": final_metrics,
            }
        )
        if not smooth_metrics["mae"] < baseline_metrics["mae"]:
            raise RuntimeError(
                "Validation Go/No-Go failed: smooth stage did not improve MAE"
            )
        if not final_metrics["mae"] < smooth_metrics["mae"]:
            raise RuntimeError(
                "Validation Go/No-Go failed: target-time stage did not improve MAE"
            )

    # Fit the frozen protocol on all Validation samples.
    val_smooth, calibrated_val, tables = fit_and_apply(
        val_prediction,
        val_truth,
        val_endpoints,
        val_prediction,
        val_endpoints,
        num_bins=args.num_bins,
        smooth_shrinkage=args.smooth_shrinkage,
        num_time_bins=args.num_time_bins,
        time_shrinkage=args.time_shrinkage,
        slices_per_day=args.slices_per_day,
    )
    centers, smooth_residuals, time_table = tables

    # Test is opened only after all Validation folds pass and tables are frozen.
    with np.load(args.predictions, allow_pickle=False) as archive:
        test_prediction = np.asarray(
            archive["test_prediction"], dtype=np.float64
        )
        test_truth = np.asarray(archive["test_truth"], dtype=np.float64)
        test_endpoints = np.asarray(archive["test_endpoints"], dtype=np.int64)
    if test_prediction.shape != test_truth.shape:
        raise ValueError("Test prediction/truth shapes differ")
    if test_prediction.shape[1:] != val_prediction.shape[1:]:
        raise ValueError("Validation/Test model output dimensions differ")
    if len(test_endpoints) != len(test_prediction):
        raise ValueError("Test endpoints are not aligned")

    test_smooth = test_prediction + args.smooth_shrinkage * smooth_correction_for(
        test_prediction, centers, smooth_residuals
    )
    calibrated_test = test_smooth + args.time_shrinkage * target_time_correction_for(
        test_endpoints,
        test_prediction.shape[1],
        time_table,
        args.slices_per_day,
    )

    baseline_val_metrics = metrics(val_prediction, val_truth)
    final_val_metrics = metrics(calibrated_val, val_truth)
    baseline_test_metrics = metrics(test_prediction, test_truth)
    smooth_test_metrics = metrics(test_smooth, test_truth)
    final_test_metrics = metrics(calibrated_test, test_truth)
    report = {
        "protocol": "validation_only_smooth_level_target_time_v1",
        "source_predictions": str(args.predictions.resolve()),
        "selection_source": "validation_only_sweep_plus_expanding_time_folds",
        "num_bins": args.num_bins,
        "smooth_shrinkage": args.smooth_shrinkage,
        "num_time_bins": args.num_time_bins,
        "time_shrinkage": args.time_shrinkage,
        "slices_per_day": args.slices_per_day,
        "target_time_definition": "(endpoint + horizon_index + 1) modulo slices_per_day",
        "validation_folds": fold_report,
        "all_validation_gates_passed": True,
        "test_opened_after_validation_gates": True,
        "baseline_val": baseline_val_metrics,
        "calibrated_val_in_sample": final_val_metrics,
        "baseline_test": baseline_test_metrics,
        "smooth_level_test": smooth_test_metrics,
        "calibrated_test": final_test_metrics,
        "test_mae_improvement": (
            baseline_test_metrics["mae"] - final_test_metrics["mae"]
        ),
        "table_values": int(
            centers.size + smooth_residuals.size + time_table.size
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with open(
        args.output_dir / "calibration.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    np.savez_compressed(
        args.output_dir / "calibrator.npz",
        num_bins=np.asarray(args.num_bins, dtype=np.int64),
        smooth_shrinkage=np.asarray(args.smooth_shrinkage, dtype=np.float64),
        num_time_bins=np.asarray(args.num_time_bins, dtype=np.int64),
        time_shrinkage=np.asarray(args.time_shrinkage, dtype=np.float64),
        slices_per_day=np.asarray(args.slices_per_day, dtype=np.int64),
        centers=centers,
        smooth_residuals=smooth_residuals,
        target_time_residuals=time_table,
    )
    np.savez_compressed(
        args.output_dir / "selected_predictions.npz",
        val_endpoints=val_endpoints,
        val_prediction=calibrated_val.astype(np.float32),
        val_truth=val_truth.astype(np.float32),
        test_endpoints=test_endpoints,
        test_prediction=calibrated_test.astype(np.float32),
        test_truth=test_truth.astype(np.float32),
    )

    # Standalone reconstruction must exactly reproduce the float32 archive.
    with np.load(
        args.output_dir / "calibrator.npz", allow_pickle=False
    ) as archive:
        saved_centers = np.asarray(archive["centers"], dtype=np.float64)
        saved_smooth_residuals = np.asarray(
            archive["smooth_residuals"], dtype=np.float64
        )
        saved_time_table = np.asarray(
            archive["target_time_residuals"], dtype=np.float64
        )
        saved_smooth_shrinkage = float(archive["smooth_shrinkage"])
        saved_time_shrinkage = float(archive["time_shrinkage"])
    reconstructed = (
        test_prediction
        + saved_smooth_shrinkage
        * smooth_correction_for(
            test_prediction, saved_centers, saved_smooth_residuals
        )
        + saved_time_shrinkage
        * target_time_correction_for(
            test_endpoints,
            test_prediction.shape[1],
            saved_time_table,
            args.slices_per_day,
        )
    ).astype(np.float32)
    with np.load(
        args.output_dir / "selected_predictions.npz", allow_pickle=False
    ) as archive:
        saved_prediction = np.asarray(archive["test_prediction"])
        saved_truth = np.asarray(archive["test_truth"])
        saved_endpoints = np.asarray(archive["test_endpoints"])
    if not np.array_equal(reconstructed, saved_prediction):
        raise AssertionError("standalone calibrator reconstruction is not exact")
    if not np.array_equal(test_truth.astype(np.float32), saved_truth):
        raise AssertionError("saved Test truth changed")
    if not np.array_equal(test_endpoints, saved_endpoints):
        raise AssertionError("saved Test endpoints changed")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
