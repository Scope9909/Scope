"""Align the frozen Hist2Res parent cache to a selected-prediction archive.

The parent cache is stored in normalized space and may contain unused trailing
rows.  This utility uses an already-audited reference archive solely for its
split endpoints and ground truth, takes the matching parent prefix, converts it
back to traffic-flow units, verifies the official metrics, and emits the common
``selected_predictions.npz`` schema consumed by the calibration scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_dualscale_quantile_residual import metrics


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=(
            root
            / "datasets/PEMS04/PEMS04_12to12_parent_predictions_joint3.npz"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=(
            root
            / "paper_results/minist_dualscale_96_720_s1/selected_predictions.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper_results/parent_aligned_for_calibration_s1",
    )
    args = parser.parse_args()

    with np.load(args.reference, allow_pickle=False) as archive:
        reference = {
            key: np.asarray(archive[key])
            for key in (
                "val_endpoints",
                "val_truth",
                "test_endpoints",
                "test_truth",
            )
        }
    with np.load(args.parent, allow_pickle=False) as archive:
        mean = float(np.asarray(archive["scaler_mean"]))
        std = float(np.asarray(archive["scaler_std"]))
        parent_z = {
            split: np.asarray(
                archive[f"teacher_pred_{split}"], dtype=np.float32
            )[..., 0]
            for split in ("val", "test")
        }

    aligned = {}
    report = {
        "parent_source": str(args.parent.resolve()),
        "reference_source": str(args.reference.resolve()),
        "scaler_mean": mean,
        "scaler_std": std,
    }
    for split in ("val", "test"):
        truth = reference[f"{split}_truth"]
        endpoints = reference[f"{split}_endpoints"]
        if len(parent_z[split]) < len(truth):
            raise ValueError(
                f"parent {split} cache is shorter than the reference: "
                f"{len(parent_z[split])} < {len(truth)}"
            )
        prediction = (
            parent_z[split][: len(truth)].astype(np.float64) * std + mean
        )
        if prediction.shape != truth.shape:
            raise ValueError(
                f"aligned {split} shape mismatch: "
                f"{prediction.shape} != {truth.shape}"
            )
        if len(endpoints) != len(prediction):
            raise ValueError(f"{split} endpoints are not aligned")
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"{split} parent prediction is non-finite")
        aligned[split] = prediction.astype(np.float32)
        report[f"{split}_samples"] = len(prediction)
        report[f"{split}_metrics"] = metrics(prediction, truth)
        report[f"{split}_unused_parent_rows"] = (
            len(parent_z[split]) - len(prediction)
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_dir / "selected_predictions.npz",
        val_endpoints=reference["val_endpoints"],
        val_prediction=aligned["val"],
        val_truth=reference["val_truth"],
        test_endpoints=reference["test_endpoints"],
        test_prediction=aligned["test"],
        test_truth=reference["test_truth"],
    )
    with open(
        args.output_dir / "alignment.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    # Verify the serialized archive, including exact labels and endpoints.
    with np.load(
        args.output_dir / "selected_predictions.npz", allow_pickle=False
    ) as archive:
        for split in ("val", "test"):
            if not np.array_equal(
                archive[f"{split}_endpoints"],
                reference[f"{split}_endpoints"],
            ):
                raise AssertionError(f"serialized {split} endpoints changed")
            if not np.array_equal(
                archive[f"{split}_truth"], reference[f"{split}_truth"]
            ):
                raise AssertionError(f"serialized {split} truth changed")
            if not np.array_equal(
                archive[f"{split}_prediction"], aligned[split]
            ):
                raise AssertionError(f"serialized {split} prediction changed")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
