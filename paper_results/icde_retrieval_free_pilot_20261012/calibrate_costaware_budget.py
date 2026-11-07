"""Fixed, leakage-safe Validation-fold calibration of Top-k vs cost-aware."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_dualscale_smooth_tod import fit_and_apply  # noqa: E402
from calibrate_dualscale_quantile_residual import metrics  # noqa: E402
from plan_cost_aware_budget import choose_ratio, gain_table, load_predictions, FOLDS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-report", type=Path, required=True)
    p.add_argument("--reference-calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.budget_report.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_calibration.read_text(encoding="utf-8"))
    no, truth, endpoints = load_predictions(Path(plan["no_memory"]))
    full, truth_full, endpoints_full = load_predictions(Path(plan["full_memory"]))
    if not np.array_equal(truth, truth_full) or not np.array_equal(endpoints, endpoints_full):
        raise ValueError("Prediction archives not aligned")
    no = no.astype(np.float64)
    full = full.astype(np.float64)
    truth = truth.astype(np.float64)
    costs = np.asarray(plan["cost_bytes_per_node"], dtype=np.int64)
    parameters = {k: reference[k] for k in
                  ("num_bins", "smooth_shrinkage", "num_time_bins",
                   "time_shrinkage", "slices_per_day")}
    results = []
    for i, (fit_fraction, eval_fraction) in enumerate(FOLDS):
        fit_end = int(len(truth) * fit_fraction)
        eval_end = int(len(truth) * eval_fraction)
        reference_fold = reference["validation_folds"][i]
        if (reference_fold["train_samples"] != fit_end or
                reference_fold["evaluation_end"] != eval_end):
            raise ValueError("Reference folds not aligned")
        gain = gain_table(no, full, truth, fit_end)
        topk = np.argsort(-gain, kind="stable")[:plan["topk_nodes"]]
        budget = int(costs[topk].sum())
        choices = {"topk": topk, "cost_aware": choose_ratio(gain, costs, budget)}
        values = {}
        for name, ids in choices.items():
            combined = no.copy()
            combined[:, :, ids] = full[:, :, ids]
            smooth, calibrated, _ = fit_and_apply(
                combined[:fit_end], truth[:fit_end], endpoints[:fit_end],
                combined[fit_end:eval_end], endpoints[fit_end:eval_end], **parameters)
            target = truth[fit_end:eval_end]
            values[name] = {
                "nodes": len(ids), "train_payload_bytes": int(costs[ids].sum()),
                "raw_mae": metrics(combined[fit_end:eval_end], target)["mae"],
                "smooth_mae": metrics(smooth, target)["mae"],
                "calibrated_mae": metrics(calibrated, target)["mae"],
            }
        results.append({
            "fit_end": fit_end, "eval_start": fit_end, "eval_end": eval_end,
            "topk": values["topk"], "cost_aware": values["cost_aware"],
            "full_memory_calibrated_mae": reference_fold["smooth_plus_target_time"]["mae"],
            "cost_aware_minus_topk_calibrated_mae":
                values["cost_aware"]["calibrated_mae"] - values["topk"]["calibrated_mae"],
        })
    report = {
        "protocol": "single_seed_fixed_validation_only_calibration_topk_vs_costaware",
        "test_evaluated": False,
        "budget_report": str(args.budget_report.resolve()),
        "reference_calibration": str(args.reference_calibration.resolve()),
        "folds": results,
        "mean_cost_aware_minus_topk_calibrated_mae": float(np.mean(
            [x["cost_aware_minus_topk_calibrated_mae"] for x in results])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
