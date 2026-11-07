"""Validation-only quantile residual calibration for dual-scale MiniST.

The calibrator is deliberately lightweight and deterministic.  For every
forecast horizon and sensor, it partitions the *prediction* values into
validation-set quantile bins and stores the median residual ``truth - pred``
inside each bin.  At inference time the frozen lookup table adds a shrunken
version of that residual to the single-model prediction.

Model selection is chronological and validation-only:

1. fit each candidate on the first ``fit_fraction`` of Validation;
2. select ``num_bins`` and ``shrinkage`` on the remaining Validation suffix;
3. refit the selected calibrator on all Validation samples;
4. only then read and evaluate Test once.

No Test label participates in fitting or hyper-parameter selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Match the repository's horizon-mean masked metric convention."""
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
        error = (
            prediction[:, horizon][mask].astype(np.float64, copy=False)
            - target
        )
        maes.append(float(np.abs(error).mean()))
        rmses.append(float(np.sqrt(np.square(error).mean())))
        mapes.append(float((np.abs(error) / target).mean()))
    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "mape": float(np.mean(mapes)),
    }


def fit_calibrator(
    prediction: np.ndarray,
    truth: np.ndarray,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-horizon/per-node prediction bins and median residuals."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("calibration arrays must have equal 3-D shapes")

    _, num_horizons, num_nodes = prediction.shape
    edges = np.empty(
        (num_bins - 1, num_horizons, num_nodes), dtype=np.float64
    )
    residuals = np.empty(
        (num_bins, num_horizons, num_nodes), dtype=np.float64
    )
    quantiles = np.arange(1, num_bins, dtype=np.float64) / num_bins

    for horizon in range(num_horizons):
        for node in range(num_nodes):
            valid = np.round(truth[:, horizon, node]) > 0
            if not np.any(valid):
                edges[:, horizon, node] = 0.0
                residuals[:, horizon, node] = 0.0
                continue
            pred = prediction[valid, horizon, node].astype(
                np.float64, copy=False
            )
            error = (
                truth[valid, horizon, node].astype(np.float64, copy=False)
                - pred
            )
            node_edges = np.quantile(pred, quantiles)
            edges[:, horizon, node] = node_edges
            bin_ids = np.searchsorted(node_edges, pred, side="right")
            fallback = float(np.median(error))
            for bin_id in range(num_bins):
                selected = error[bin_ids == bin_id]
                residuals[bin_id, horizon, node] = (
                    float(np.median(selected)) if selected.size else fallback
                )

    if not np.isfinite(edges).all() or not np.isfinite(residuals).all():
        raise FloatingPointError("calibrator contains NaN or infinite values")
    return edges, residuals


def correction_for(
    prediction: np.ndarray,
    edges: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """Look up the frozen residual correction for each prediction."""
    if prediction.ndim != 3:
        raise ValueError("prediction must have shape [sample,horizon,node]")
    num_bins, num_horizons, num_nodes = residuals.shape
    if edges.shape != (num_bins - 1, num_horizons, num_nodes):
        raise ValueError("edge and residual table shapes are inconsistent")
    if prediction.shape[1:] != (num_horizons, num_nodes):
        raise ValueError("prediction shape does not match calibrator")

    correction = np.empty_like(prediction, dtype=np.float64)
    for horizon in range(num_horizons):
        for node in range(num_nodes):
            bin_ids = np.searchsorted(
                edges[:, horizon, node],
                prediction[:, horizon, node],
                side="right",
            )
            correction[:, horizon, node] = residuals[
                bin_ids, horizon, node
            ]
    if not np.isfinite(correction).all():
        raise FloatingPointError("calibration correction is not finite")
    return correction


def parse_csv(raw: str, converter):
    values = [converter(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("candidate list must not be empty")
    return values


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
        default=(
            root
            / "paper_results/minist_dualscale_quantile_calibration_s1"
        ),
    )
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--bin-candidates", default="2,3,4,6,8")
    parser.add_argument("--shrinkage-candidates", default="0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    if not 0.5 <= args.fit_fraction < 1.0:
        raise ValueError("fit_fraction must be in [0.5, 1.0)")
    bin_candidates = parse_csv(args.bin_candidates, int)
    shrinkage_candidates = parse_csv(args.shrinkage_candidates, float)
    if any(value < 2 for value in bin_candidates):
        raise ValueError("all bin candidates must be at least 2")
    if any(not 0.0 < value <= 1.0 for value in shrinkage_candidates):
        raise ValueError("all shrinkage candidates must be in (0, 1]")

    # Read Validation only while selecting the calibration protocol.
    with np.load(args.predictions, allow_pickle=False) as archive:
        val_prediction = np.asarray(archive["val_prediction"], dtype=np.float64)
        val_truth = np.asarray(archive["val_truth"], dtype=np.float64)
        val_endpoints = np.asarray(archive["val_endpoints"])
    if val_prediction.shape != val_truth.shape:
        raise ValueError("Validation prediction/truth shapes do not match")
    if len(val_endpoints) != len(val_prediction):
        raise ValueError("Validation endpoints are not aligned")
    if not np.isfinite(val_prediction).all() or not np.isfinite(val_truth).all():
        raise FloatingPointError("Validation arrays contain NaN or infinity")

    split_index = int(len(val_prediction) * args.fit_fraction)
    if split_index <= 0 or split_index >= len(val_prediction):
        raise ValueError("fit_fraction creates an empty selection partition")
    fit_prediction, select_prediction = (
        val_prediction[:split_index],
        val_prediction[split_index:],
    )
    fit_truth, select_truth = val_truth[:split_index], val_truth[split_index:]

    candidates = []
    for num_bins in bin_candidates:
        edges, residuals = fit_calibrator(
            fit_prediction, fit_truth, num_bins
        )
        correction = correction_for(select_prediction, edges, residuals)
        for shrinkage in shrinkage_candidates:
            candidate_metrics = metrics(
                select_prediction + shrinkage * correction,
                select_truth,
            )
            candidates.append(
                {
                    "num_bins": int(num_bins),
                    "shrinkage": float(shrinkage),
                    "selection": candidate_metrics,
                }
            )

    selected = min(
        candidates,
        key=lambda item: (
            item["selection"]["mae"],
            item["num_bins"],
            item["shrinkage"],
        ),
    )
    selected_bins = selected["num_bins"]
    selected_shrinkage = selected["shrinkage"]

    # Refit the frozen, Validation-selected design on all Validation samples.
    edges, residuals = fit_calibrator(
        val_prediction, val_truth, selected_bins
    )
    val_correction = correction_for(val_prediction, edges, residuals)
    calibrated_val = val_prediction + selected_shrinkage * val_correction

    # Test is intentionally opened only after model selection and full-Val fit.
    with np.load(args.predictions, allow_pickle=False) as archive:
        test_prediction = np.asarray(
            archive["test_prediction"], dtype=np.float64
        )
        test_truth = np.asarray(archive["test_truth"], dtype=np.float64)
        test_endpoints = np.asarray(archive["test_endpoints"])
    if test_prediction.shape != test_truth.shape:
        raise ValueError("Test prediction/truth shapes do not match")
    if test_prediction.shape[1:] != val_prediction.shape[1:]:
        raise ValueError("Validation/Test horizon or node counts differ")
    if len(test_endpoints) != len(test_prediction):
        raise ValueError("Test endpoints are not aligned")
    if not np.isfinite(test_prediction).all() or not np.isfinite(test_truth).all():
        raise FloatingPointError("Test arrays contain NaN or infinity")

    test_correction = correction_for(test_prediction, edges, residuals)
    calibrated_test = (
        test_prediction + selected_shrinkage * test_correction
    )
    baseline_val = metrics(val_prediction, val_truth)
    calibrated_val_metrics = metrics(calibrated_val, val_truth)
    baseline_test = metrics(test_prediction, test_truth)
    calibrated_test_metrics = metrics(calibrated_test, test_truth)

    report = {
        "protocol": "chronological_validation_only_quantile_residual_v1",
        "source_predictions": str(args.predictions.resolve()),
        "fit_fraction": args.fit_fraction,
        "fit_samples": split_index,
        "selection_samples": len(val_prediction) - split_index,
        "bin_candidates": bin_candidates,
        "shrinkage_candidates": shrinkage_candidates,
        "selection_baseline": metrics(select_prediction, select_truth),
        "candidates": candidates,
        "selected_num_bins": selected_bins,
        "selected_shrinkage": selected_shrinkage,
        "selected_metrics": selected["selection"],
        "test_opened_after_selection": True,
        "baseline_val": baseline_val,
        "calibrated_val": calibrated_val_metrics,
        "baseline_test": baseline_test,
        "calibrated_test": calibrated_test_metrics,
        "test_mae_improvement": (
            baseline_test["mae"] - calibrated_test_metrics["mae"]
        ),
        "table_parameters": int(edges.size + residuals.size),
        "mean_abs_test_correction": float(
            np.abs(selected_shrinkage * test_correction).mean()
        ),
        "max_abs_test_correction": float(
            np.abs(selected_shrinkage * test_correction).max()
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with open(
        args.output_dir / "calibration.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    np.savez_compressed(
        args.output_dir / "calibrator.npz",
        num_bins=np.asarray(selected_bins, dtype=np.int64),
        # Keep lookup boundaries in float64.  Casting quantile edges to
        # float32 can move a prediction lying exactly next to an edge into a
        # different bin and therefore break standalone reconstruction.
        shrinkage=np.asarray(selected_shrinkage, dtype=np.float64),
        edges=edges,
        median_residuals=residuals,
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

    # Verify both the serialized predictions and standalone frozen calibrator.
    with np.load(
        args.output_dir / "selected_predictions.npz", allow_pickle=False
    ) as archive:
        saved_prediction = np.asarray(archive["test_prediction"])
        saved_truth = np.asarray(archive["test_truth"])
        saved_metrics = metrics(
            saved_prediction,
            saved_truth,
        )
    if abs(saved_metrics["mae"] - calibrated_test_metrics["mae"]) > 1e-5:
        raise AssertionError("serialized prediction MAE failed verification")
    with np.load(
        args.output_dir / "calibrator.npz", allow_pickle=False
    ) as archive:
        saved_shrinkage = float(np.asarray(archive["shrinkage"]))
        saved_edges = np.asarray(archive["edges"], dtype=np.float64)
        saved_residuals = np.asarray(
            archive["median_residuals"], dtype=np.float64
        )
    reconstructed = test_prediction + saved_shrinkage * correction_for(
        test_prediction, saved_edges, saved_residuals
    )
    # The prediction archive is intentionally float32.  Compare in that
    # serialized dtype so ordinary float32 rounding (up to roughly 3e-5 for
    # large traffic values) is not mistaken for a bin-assignment mismatch.
    reconstructed_serialized = reconstructed.astype(saved_prediction.dtype)
    reconstruction_error = float(
        np.max(np.abs(reconstructed_serialized - saved_prediction))
    )
    if reconstruction_error != 0.0:
        raise AssertionError(
            "serialized calibrator reconstruction failed: "
            f"max_abs={reconstruction_error}"
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
