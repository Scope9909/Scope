"""Fixed-parameter ValCal audit of a 60%-sensor sparse-memory pilot."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-nodes", type=int, default=184)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    pilot_root = ROOT / "paper_results" / "icde_retrieval_free_pilot_20261012"
    results = []
    for seed in (1, 2, 3):
        with np.load(
            pilot_root / f"node_budget_s{seed}_no_memory" / "no_memory_validation_predictions.npz",
            allow_pickle=False,
        ) as archive:
            no_memory = np.asarray(archive["val_prediction"], dtype=np.float64)
            truth = np.asarray(archive["val_truth"], dtype=np.float64)
            endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
        with np.load(
            ROOT / "paper_results" / "icde_generality" /
            f"pems04_joint3_lite_r256_strictalign_s{seed}" / "selected_predictions.npz",
            allow_pickle=False,
        ) as archive:
            full_memory = np.asarray(archive["val_prediction"], dtype=np.float64)
            if not np.array_equal(truth, archive["val_truth"]):
                raise ValueError(f"Seed {seed} truth mismatch")
            if not np.array_equal(endpoints, archive["val_endpoints"]):
                raise ValueError(f"Seed {seed} endpoints mismatch")
        reference_path = (
            ROOT / "paper_results" / "icde_generality" /
            f"pems04_joint3_lite_r256_strictalign_s{seed}_valcal" / "calibration.json"
        )
        with reference_path.open("r", encoding="utf-8") as stream:
            reference = json.load(stream)
        parameters = {
            key: reference[key]
            for key in (
                "num_bins", "smooth_shrinkage", "num_time_bins",
                "time_shrinkage", "slices_per_day",
            )
        }
        folds = []
        n = len(truth)
        for fold_index, (train_fraction, end_fraction) in enumerate(
            ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0))
        ):
            train_end = int(train_fraction * n)
            eval_end = int(end_fraction * n)
            reference_fold = reference["validation_folds"][fold_index]
            if (
                reference_fold["train_samples"] != train_end
                or reference_fold["evaluation_end"] != eval_end
            ):
                raise ValueError("Calibration reference folds are not aligned")
            valid = np.round(truth[:train_end]) > 0
            denominator = valid.sum(axis=(0, 1))
            gain = (
                (np.abs(no_memory[:train_end] - truth[:train_end]) * valid).sum(axis=(0, 1))
                - (np.abs(full_memory[:train_end] - truth[:train_end]) * valid).sum(axis=(0, 1))
            ) / denominator
            selected = np.argsort(-gain, kind="stable")[:args.selected_nodes]
            combined = no_memory.copy()
            combined[:, :, selected] = full_memory[:, :, selected]
            smooth, calibrated, _ = fit_and_apply(
                combined[:train_end],
                truth[:train_end],
                endpoints[:train_end],
                combined[train_end:eval_end],
                endpoints[train_end:eval_end],
                **parameters,
            )
            target = truth[train_end:eval_end]
            raw = metrics(combined[train_end:eval_end], target)
            smooth_metrics = metrics(smooth, target)
            calibrated_metrics = metrics(calibrated, target)
            reference_calibrated = reference_fold["smooth_plus_target_time"]
            fold = {
                "selection_samples": train_end,
                "evaluation_start": train_end,
                "evaluation_end": eval_end,
                "raw": raw,
                "smooth": smooth_metrics,
                "calibrated": calibrated_metrics,
                "full_memory_calibrated": reference_calibrated,
                "calibrated_mae_minus_full_memory": (
                    calibrated_metrics["mae"] - reference_calibrated["mae"]
                ),
                "smooth_gate_passed": smooth_metrics["mae"] < raw["mae"],
                "target_time_gate_passed": calibrated_metrics["mae"] < smooth_metrics["mae"],
            }
            folds.append(fold)
            print(
                f"Seed {seed} fold {fold_index + 1}: sparse calibrated MAE "
                f"{calibrated_metrics['mae']:.4f}; versus full "
                f"{fold['calibrated_mae_minus_full_memory']:+.4f}",
                flush=True,
            )
        results.append({"seed": seed, "folds": folds})
    differences = [
        fold["calibrated_mae_minus_full_memory"]
        for item in results for fold in item["folds"]
    ]
    report = {
        "protocol": "PEMS04_60percent_sensor_sparse_memory_fixed_ValCal_three_seeds_three_folds",
        "selected_nodes": args.selected_nodes,
        "selection_and_calibration_fit_on_prior_Validation_only": True,
        "test_evaluated": False,
        "seeds_2_and_3_sparse_predictions_are_algebraic_sensor_mixes": True,
        "seed_1_sparse_forward_algebraic_equivalence_audited_separately": True,
        "folds": results,
        "summary": {
            "calibrated_mae_minus_full_memory_mean": float(np.mean(differences)),
            "calibrated_mae_minus_full_memory_sample_std": float(np.std(differences, ddof=1)),
            "improved_fold_seed_count": int(np.sum(np.asarray(differences) < 0)),
            "total_fold_seed_count": len(differences),
            "all_smooth_gates_passed": all(
                fold["smooth_gate_passed"] for item in results for fold in item["folds"]
            ),
            "all_target_time_gates_passed": all(
                fold["target_time_gate_passed"] for item in results for fold in item["folds"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
