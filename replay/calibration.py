"""Validation-fitted level and target-time calibration used by frozen replay."""
from __future__ import annotations

import numpy as np


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compute horizon-mean metrics over targets satisfying round(y) > 0."""
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError(
            f"expected equal [sample,horizon,node] arrays, got "
            f"{prediction.shape} and {truth.shape}"
        )
    maes, rmses, mapes = [], [], []
    for horizon in range(prediction.shape[1]):
        mask = np.round(truth[:, horizon]) > 0
        if not np.any(mask):
            raise ValueError(f"horizon {horizon} contains no valid targets")
        target = truth[:, horizon][mask].astype(np.float64, copy=False)
        error = prediction[:, horizon][mask].astype(np.float64, copy=False) - target
        maes.append(float(np.abs(error).mean()))
        rmses.append(float(np.sqrt(np.square(error).mean())))
        mapes.append(float((np.abs(error) / target).mean()))
    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "mape": float(np.mean(mapes)),
    }


def fit_smooth_level(prediction, truth, num_bins):
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and truth must be equal 3-D arrays")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    _, num_horizons, num_nodes = prediction.shape
    centers = np.empty((num_bins, num_horizons, num_nodes), dtype=np.float64)
    residuals = np.empty_like(centers)
    edge_quantiles = np.arange(1, num_bins, dtype=np.float64) / num_bins
    center_quantiles = (np.arange(num_bins, dtype=np.float64) + 0.5) / num_bins
    for horizon in range(num_horizons):
        for node in range(num_nodes):
            valid = np.round(truth[:, horizon, node]) > 0
            if not np.any(valid):
                centers[:, horizon, node] = 0.0
                residuals[:, horizon, node] = 0.0
                continue
            pred = prediction[valid, horizon, node].astype(np.float64, copy=False)
            error = truth[valid, horizon, node].astype(np.float64, copy=False) - pred
            edges = np.quantile(pred, edge_quantiles)
            bin_ids = np.searchsorted(edges, pred, side="right")
            fallback = float(np.median(error))
            for bin_id in range(num_bins):
                selected = bin_ids == bin_id
                if np.any(selected):
                    centers[bin_id, horizon, node] = float(np.median(pred[selected]))
                    residuals[bin_id, horizon, node] = float(np.median(error[selected]))
                else:
                    centers[bin_id, horizon, node] = float(
                        np.quantile(pred, center_quantiles[bin_id])
                    )
                    residuals[bin_id, horizon, node] = fallback
    if not np.isfinite(centers).all() or not np.isfinite(residuals).all():
        raise FloatingPointError("smooth level table contains non-finite values")
    return centers, residuals


def smooth_correction_for(prediction, centers, residuals):
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
                    [np.median(values[inverse == index]) for index in range(len(unique_x))],
                    dtype=np.float64,
                )
                x = unique_x
            correction[:, horizon, node] = np.interp(
                prediction[:, horizon, node], x, values,
                left=values[0], right=values[-1],
            )
    if not np.isfinite(correction).all():
        raise FloatingPointError("smooth correction contains non-finite values")
    return correction


def target_time_bucket_ids(endpoints, num_horizons, slices_per_day, num_time_bins):
    if slices_per_day % num_time_bins != 0:
        raise ValueError("num_time_bins must divide slices_per_day exactly")
    target_offsets = 1 + np.arange(num_horizons, dtype=np.int64)
    target_slots = (
        endpoints.astype(np.int64)[:, None] + target_offsets[None, :]
    ) % slices_per_day
    return target_slots * num_time_bins // slices_per_day


def fit_target_time_residual(prediction, truth, endpoints, slices_per_day, num_time_bins):
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
            table[time_bin, node] = float(np.median(selected)) if selected.size else 0.0
    if not np.isfinite(table).all():
        raise FloatingPointError("target-time residual table is non-finite")
    return table


def target_time_correction_for(endpoints, num_horizons, table, slices_per_day):
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
    fit_prediction, fit_truth, fit_endpoints, query_prediction, query_endpoints,
    *, num_bins, smooth_shrinkage, num_time_bins, time_shrinkage, slices_per_day,
):
    centers, smooth_residuals = fit_smooth_level(fit_prediction, fit_truth, num_bins)
    fit_smooth = fit_prediction + smooth_shrinkage * smooth_correction_for(
        fit_prediction, centers, smooth_residuals
    )
    query_smooth = query_prediction + smooth_shrinkage * smooth_correction_for(
        query_prediction, centers, smooth_residuals
    )
    time_table = fit_target_time_residual(
        fit_smooth, fit_truth, fit_endpoints, slices_per_day, num_time_bins
    )
    query_final = query_smooth + time_shrinkage * target_time_correction_for(
        query_endpoints, query_prediction.shape[1], time_table, slices_per_day
    )
    return query_smooth, query_final, (centers, smooth_residuals, time_table)
