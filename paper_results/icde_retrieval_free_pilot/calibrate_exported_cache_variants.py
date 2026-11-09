"""Compare one checkpoint's cache variants on chronological Validation folds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibrate_dualscale_smooth_tod import fit_and_apply  # noqa: E402
from calibrate_dualscale_quantile_residual import metrics  # noqa: E402

VARIANTS = ("fp32_full", "fp32_sparse", "fp16_full", "int8_full",
            "fp16_sparse", "int8_sparse")
FOLDS = ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant-dir", type=Path, required=True)
    p.add_argument("--reference-calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", action="append", nargs=2, metavar=("NAME", "PATH"),
                   help="Optional explicit variant pair; overrides the six default variants")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads(args.reference_calibration.read_text(encoding="utf-8"))
    parameters = {key: reference[key] for key in
                  ("num_bins", "smooth_shrinkage", "num_time_bins",
                   "time_shrinkage", "slices_per_day")}
    report = {"protocol": "single_checkpoint_six_cache_variants_validation_only_fixed_calibration",
              "test_evaluated": False,
              "selection": "sparse nodes fixed from first 55% of Validation",
              "reference_calibration": str(args.reference_calibration.resolve()),
              "variants": {}}
    truth_ref = endpoints_ref = None
    variants = [(name, args.variant_dir / f"{name}_val_predictions.npz")
                for name in VARIANTS] if not args.variant else [
                    (name, Path(path)) for name, path in args.variant]
    for name, path in variants:
        with np.load(path, allow_pickle=False) as archive:
            prediction = np.asarray(archive["val_prediction"], dtype=np.float64)
            truth = np.asarray(archive["val_truth"], dtype=np.float64)
            endpoints = np.asarray(archive["val_endpoints"])
        if truth_ref is None:
            truth_ref, endpoints_ref = truth, endpoints
        elif not np.array_equal(truth, truth_ref) or not np.array_equal(endpoints, endpoints_ref):
            raise ValueError(f"{name} targets/endpoints differ")
        folds = []
        for i, (fit_fraction, eval_fraction) in enumerate(FOLDS):
            fit_end = int(len(truth) * fit_fraction)
            eval_end = int(len(truth) * eval_fraction)
            ref_fold = reference["validation_folds"][i]
            if (ref_fold["train_samples"] != fit_end or
                    ref_fold["evaluation_end"] != eval_end):
                raise ValueError("Reference fold boundaries differ")
            smooth, calibrated, _ = fit_and_apply(
                prediction[:fit_end], truth[:fit_end], endpoints[:fit_end],
                prediction[fit_end:eval_end], endpoints[fit_end:eval_end],
                **parameters)
            target = truth[fit_end:eval_end]
            raw = metrics(prediction[fit_end:eval_end], target)
            smooth_metric = metrics(smooth, target)
            final = metrics(calibrated, target)
            folds.append({
                "fit_end": fit_end, "eval_start": fit_end, "eval_end": eval_end,
                "raw_mae": raw["mae"], "smooth_mae": smooth_metric["mae"],
                "calibrated_mae": final["mae"],
                "smooth_gate": smooth_metric["mae"] < raw["mae"],
                "target_time_gate": final["mae"] < smooth_metric["mae"],
            })
            print(f"{name} fold {i+1}: raw {raw['mae']:.6f}, calibrated {final['mae']:.6f}",
                  flush=True)
        report["variants"][name] = {
            "prediction_path": str(path.resolve()), "folds": folds,
            "mean_raw_mae": float(np.mean([f["raw_mae"] for f in folds])),
            "mean_calibrated_mae": float(np.mean([f["calibrated_mae"] for f in folds])),
            "all_gates_passed": all(f["smooth_gate"] and f["target_time_gate"]
                                    for f in folds),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in item.items()
                             if key in ("mean_raw_mae", "mean_calibrated_mae", "all_gates_passed")}
                      for name, item in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
