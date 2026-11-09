"""Paired moving-block uncertainty for the final Validation fold.

Re-fits the already frozen calibration rule on the preceding Validation
prefix. This is descriptive uncertainty, not a new model-selection gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibrate_dualscale_smooth_tod import fit_and_apply  # noqa: E402


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(np.asarray(archive[key]) for key in
                     ("val_prediction", "val_truth", "val_endpoints"))


def corrected(path, reference, fit_end, eval_end):
    prediction, truth, endpoints = load(path)
    params = {key: reference[key] for key in
              ("num_bins", "smooth_shrinkage", "num_time_bins",
               "time_shrinkage", "slices_per_day")}
    _, calibrated, _ = fit_and_apply(
        prediction[:fit_end].astype(np.float64),
        truth[:fit_end].astype(np.float64), endpoints[:fit_end],
        prediction[fit_end:eval_end].astype(np.float64),
        endpoints[fit_end:eval_end], **params)
    return calibrated, truth[fit_end:eval_end], endpoints[fit_end:eval_end]


def bootstrap(numerator, denominator, block, trials, rng):
    n = len(numerator)
    blocks = (n + block - 1) // block
    stats = np.empty(trials, dtype=np.float64)
    for i in range(trials):
        starts = rng.integers(0, n - block + 1, size=blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
        stats[i] = numerator[idx].sum() / denominator[idx].sum()
    return {"block_samples": block, "trials": trials,
            "ci_95": np.quantile(stats, [0.025, 0.975]).tolist(),
            "probability_benefit": float(np.mean(stats < 0))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topk-predictions", type=Path, required=True)
    p.add_argument("--budget-predictions", type=Path, required=True)
    p.add_argument("--reference-calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=2000)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads(args.reference_calibration.read_text(encoding="utf-8"))
    fold = reference["validation_folds"][-1]
    fit_end, eval_end = fold["train_samples"], fold["evaluation_end"]
    topk, truth, endpoints = corrected(args.topk_predictions, reference, fit_end, eval_end)
    budget, truth2, endpoints2 = corrected(args.budget_predictions, reference, fit_end, eval_end)
    if not np.array_equal(truth, truth2) or not np.array_equal(endpoints, endpoints2):
        raise ValueError("Prediction archives are not aligned")
    valid = np.round(truth) > 0
    numerator = ((np.abs(budget - truth) - np.abs(topk - truth)) * valid).sum(axis=(1, 2))
    denominator = valid.sum(axis=(1, 2))
    observed = float(numerator.sum() / denominator.sum())
    rng = np.random.default_rng(20261013)
    block_results = [bootstrap(numerator, denominator, block, args.trials, rng)
                     for block in (12, 72, 288) if block <= len(numerator)]
    report = {"protocol": "last_validation_fold_paired_moving_block_bootstrap",
              "test_evaluated": False, "metric": "budget_minus_topk_calibrated_mae",
              "fit_end": fit_end, "eval_end": eval_end,
              "observed_delta": observed, "blocks": block_results,
              "interpretation": "Descriptive, sensitive to block length; not independent test evidence"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
