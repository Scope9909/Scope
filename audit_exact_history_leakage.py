"""Deterministic leakage and coverage audit for exact-history caches.

Validation and test caches are fully recomputed using only observable query
histories and training-split candidate histories/futures.  A deterministic
training subset is also recomputed with leave-one-out enabled.  The script
then reports exact-hit coverage and frozen-model errors on hit/non-hit cells.
No parameter is trained or selected.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent

DATASETS = {
    "PEMS03": {
        "cache": "datasets/PEMS03/PEMS03_12to12_analog_k4_exact_mean.npz",
        "index": "datasets/PEMS03/PEMS03_12to12_index_scaler.npz",
        "network": "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz",
        "full": "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz",
        "teacher": "datasets/PEMS03/PEMS03_12to12_parent_predictions_joint3.npz",
    },
    "PEMS04": {
        "cache": "datasets/PEMS04/PEMS04_12to12_analog_k4_exact_mean.npz",
        "index": "datasets/PEMS04/PEMS04_12to12_index_scaler.npz",
        "network": "paper_results/lite_r256_kdfix3_a30_lr1e4_aligned_s1/selected_predictions.npz",
        "full": "paper_results/lite_r256_kdfix3_a30_lr1e4_smooth_tod_s1/selected_predictions.npz",
        "teacher": "datasets/PEMS04/PEMS04_12to12_parent_predictions_joint3.npz",
    },
    "PEMS07": {
        "cache": "datasets/PEMS07/PEMS07_12to12_analog_k4_exact_mean.npz",
        "index": "datasets/PEMS07/PEMS07_12to12_index_scaler.npz",
        "network": "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz",
        "full": "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz",
        "teacher": "datasets/PEMS07/PEMS07_12to12_parent_predictions_joint3.npz",
    },
    "PEMS08": {
        "cache": "datasets/PEMS08/PEMS08_12to12_analog_k4_exact_mean.npz",
        "index": "datasets/PEMS08/PEMS08_12to12_index_scaler.npz",
        "network": "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz",
        "full": "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz",
        "teacher": "datasets/PEMS08/PEMS08_12to12_parent_predictions_joint3.npz",
    },
    "PEMSBAY": {
        "cache": "datasets/PEMSBAY/PEMSBAY_12to12_analog_k8_exact_mean_712.npz",
        "index": "datasets/PEMSBAY/PEMSBAY_12to12_index_scaler_712.npz",
        "network": "paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz",
        "full": "paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz",
        "teacher": "datasets/PEMSBAY/PEMSBAY_12to12_parent_predictions_final_joint_712.npz",
    },
}


def normalize(values, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    if mean.size == 1:
        return ((values - float(mean.reshape(-1)[0])) / float(std.reshape(-1)[0])).astype(np.float32)
    return ((values - mean.reshape(1, 1, -1, 1)) / std.reshape(1, 1, -1, 1)).astype(np.float32)


def recompute_exact(data, time_data, query_indices, train_indices, min_valid, leave_one_out):
    x_offsets = np.arange(-11, 1, dtype=np.int64)
    y_offsets = np.arange(1, 13, dtype=np.int64)
    forecast = np.zeros((len(query_indices), 12, data.shape[1], 1), dtype=np.float32)
    available = np.zeros_like(forecast, dtype=np.uint8)
    query_tod = time_data[query_indices, 0]
    train_tod = time_data[train_indices, 0]
    for tod in np.unique(query_tod):
        positions = np.flatnonzero(query_tod == tod)
        queries = query_indices[positions]
        candidates = train_indices[train_tod == tod]
        qhist = data[queries[:, None] + x_offsets[None, :], :, 0].astype(np.float32, copy=False)
        chist = data[candidates[:, None] + x_offsets[None, :], :, 0].astype(np.float32, copy=False)
        valid = (qhist[:, None] != 0) & (chist[None] != 0)
        difference = np.where(valid, qhist[:, None] - chist[None], 0.0)
        distance = np.einsum("qctn,qctn->qcn", difference, difference, optimize=True)
        exact = (valid.sum(axis=2) >= min_valid) & (distance == 0)
        if leave_one_out:
            location = {int(value): pos for pos, value in enumerate(candidates)}
            own = np.asarray([location[int(value)] for value in queries], dtype=np.int64)
            exact[np.arange(len(queries)), own, :] = False
        future = data[candidates[:, None] + y_offsets[None, :], :, 0].astype(np.float32, copy=False).transpose(0, 2, 1)
        selected = exact[..., None] & (future[None] != 0)
        count = selected.sum(axis=1)
        total = np.where(selected, future[None], 0.0).sum(axis=1, dtype=np.float64)
        group = np.zeros_like(total, dtype=np.float64)
        np.divide(total, count, out=group, where=count > 0)
        forecast[positions, ..., 0] = group.transpose(0, 2, 1)
        available[positions, ..., 0] = (count > 0).transpose(0, 2, 1).astype(np.uint8)
    return forecast, available


def load_teacher(path, count):
    with np.load(path, allow_pickle=False) as archive:
        prediction = np.asarray(archive["teacher_pred_test"][:count], dtype=np.float64)
        mean = np.asarray(archive["scaler_mean"], dtype=np.float64)
        std = np.asarray(archive["scaler_std"], dtype=np.float64)
    if prediction.ndim == 4:
        prediction = prediction[..., 0]
    if mean.size == 1:
        return prediction * float(std.reshape(-1)[0]) + float(mean.reshape(-1)[0])
    return prediction * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def group_mae(prediction, truth, group):
    mask = np.isfinite(truth) & (truth != 0) & group
    return float(np.abs(prediction - truth)[mask].mean()) if np.any(mask) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-audit-samples", type=int, default=64)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "exact_history_train_only_full_val_test_recompute_v1",
        "selection_inputs": "query history only; candidate histories restricted to idx_train at the same time-of-day",
        "candidate_target_access": "only candidate futures indexed by idx_train are read after exact history selection",
        "validation_test_future_used_for_selection": False,
        "datasets": {},
    }
    csv_rows = []

    for dataset, paths in DATASETS.items():
        data_path = ROOT / f"datasets/{dataset}/data.npz"
        time_path = ROOT / f"datasets/{dataset}/{dataset}_time.npz"
        with np.load(data_path, allow_pickle=False) as archive:
            data = np.asarray(archive["data"])
        with np.load(time_path, allow_pickle=False) as archive:
            time_data = np.asarray(archive["time"])
        with np.load(ROOT / paths["index"], allow_pickle=False) as archive:
            active = {name: np.asarray(archive[name], dtype=np.int64) for name in ("idx_train", "idx_val", "idx_test")}
            active_mean = np.asarray(archive["scaler_mean"])
            active_std = np.asarray(archive["scaler_std"])

        with np.load(ROOT / paths["cache"], allow_pickle=False) as cache:
            cached_indices = {name: np.asarray(cache[name], dtype=np.int64) for name in active}
            split_match = {name: bool(np.array_equal(active[name], cached_indices[name])) for name in active}
            metadata = {
                "exact_match_rule": str(np.asarray(cache["exact_match_rule"]).item()),
                "exact_min_valid_history": int(np.asarray(cache["exact_min_valid_history"]).item()),
                "train_leave_one_out": int(np.asarray(cache["train_leave_one_out"]).item()),
                "exact_train_leave_one_out": int(np.asarray(cache["exact_train_leave_one_out"]).item()),
            }
            scaler_match = bool(
                np.array_equal(np.asarray(cache["scaler_mean"]), active_mean)
                and np.array_equal(np.asarray(cache["scaler_std"]), active_std)
            )
            disjoint = bool(
                not np.intersect1d(active["idx_train"], active["idx_val"]).size
                and not np.intersect1d(active["idx_train"], active["idx_test"]).size
                and not np.intersect1d(active["idx_val"], active["idx_test"]).size
            )
            chronological = bool(
                active["idx_train"].max() < active["idx_val"].min()
                and active["idx_val"].max() < active["idx_test"].min()
            )

            recompute = {}
            for split in ("val", "test"):
                raw, available = recompute_exact(
                    data, time_data, active[f"idx_{split}"], active["idx_train"],
                    metadata["exact_min_valid_history"], False,
                )
                rebuilt = normalize(raw, active_mean, active_std)
                stored_prediction = np.asarray(cache[f"exact_retrieval_{split}"])
                stored_available = np.asarray(cache[f"exact_available_{split}"], dtype=np.uint8)
                recompute[split] = {
                    "availability_bitwise_equal": bool(np.array_equal(available, stored_available)),
                    "prediction_max_abs_error": float(np.max(np.abs(rebuilt.astype(np.float32) - stored_prediction.astype(np.float32)))),
                    "prediction_allclose": bool(np.allclose(rebuilt, stored_prediction, rtol=1e-5, atol=2e-4)),
                }
                del raw, rebuilt, stored_prediction

            train_positions = np.unique(np.linspace(
                0, len(active["idx_train"]) - 1,
                min(args.train_audit_samples, len(active["idx_train"])), dtype=np.int64,
            ))
            train_queries = active["idx_train"][train_positions]
            raw_train, available_train = recompute_exact(
                data, time_data, train_queries, active["idx_train"],
                metadata["exact_min_valid_history"], True,
            )
            rebuilt_train = normalize(raw_train, active_mean, active_std)
            stored_train = np.asarray(cache["exact_retrieval_train"])[train_positions]
            stored_train_available = np.asarray(cache["exact_available_train"], dtype=np.uint8)[train_positions]
            train_recompute = {
                "samples": int(len(train_positions)),
                "leave_one_out_enabled": True,
                "availability_bitwise_equal": bool(np.array_equal(available_train, stored_train_available)),
                "prediction_max_abs_error": float(np.max(np.abs(rebuilt_train.astype(np.float32) - stored_train.astype(np.float32)))),
                "prediction_allclose": bool(np.allclose(rebuilt_train, stored_train, rtol=1e-5, atol=2e-4)),
            }
            exact_test = np.asarray(cache["exact_available_test"], dtype=bool)[..., 0]

        with np.load(ROOT / paths["network"], allow_pickle=False) as archive:
            endpoints = np.asarray(archive["test_endpoints"], dtype=np.int64)
            network = np.asarray(archive["test_prediction"], dtype=np.float64)
            truth = np.asarray(archive["test_truth"], dtype=np.float64)
        with np.load(ROOT / paths["full"], allow_pickle=False) as archive:
            full_endpoints = np.asarray(archive["test_endpoints"], dtype=np.int64)
            full = np.asarray(archive["test_prediction"], dtype=np.float64)
            full_truth = np.asarray(archive["test_truth"], dtype=np.float64)
        endpoint_alignment = bool(
            np.array_equal(endpoints, active["idx_test"])
            and np.array_equal(full_endpoints, active["idx_test"])
        )
        truth_alignment = bool(np.array_equal(truth, full_truth))
        teacher = load_teacher(ROOT / paths["teacher"], len(truth))
        valid_truth = np.isfinite(truth) & (truth != 0)
        hit = exact_test & valid_truth
        miss = (~exact_test) & valid_truth
        coverage = {
            "valid_elements": int(valid_truth.sum()),
            "exact_hit_elements": int(hit.sum()),
            "element_coverage_percent": float(100 * hit.sum() / valid_truth.sum()),
            "sample_any_hit_percent": float(100 * exact_test.reshape(len(exact_test), -1).any(axis=1).mean()),
        }
        errors = {}
        for name, prediction in (("teacher", teacher), ("network_only", network), ("full_system", full)):
            errors[name] = {
                "exact_hit_mae_element_weighted": group_mae(prediction, truth, hit),
                "non_hit_mae_element_weighted": group_mae(prediction, truth, miss),
                "overall_mae_element_weighted": group_mae(prediction, truth, valid_truth),
            }
        all_pass = bool(
            all(split_match.values()) and scaler_match and disjoint and chronological
            and metadata["exact_match_rule"] == "masked_mse_eq_0"
            and metadata["train_leave_one_out"] == 1
            and metadata["exact_train_leave_one_out"] == 1
            and all(item["availability_bitwise_equal"] and item["prediction_allclose"] for item in recompute.values())
            and train_recompute["availability_bitwise_equal"] and train_recompute["prediction_allclose"]
            and endpoint_alignment and truth_alignment
        )
        record = {
            "cache": str((ROOT / paths["cache"]).resolve()),
            "split_arrays_match_active_index": split_match,
            "scaler_matches_active_index": scaler_match,
            "split_endpoints_disjoint": disjoint,
            "split_order_is_chronological": chronological,
            "metadata": metadata,
            "full_recompute": recompute,
            "train_leave_one_out_recompute": train_recompute,
            "prediction_archive_endpoint_alignment": endpoint_alignment,
            "prediction_archive_truth_alignment": truth_alignment,
            "coverage": coverage,
            "grouped_test_errors": errors,
            "audit_pass": all_pass,
        }
        report["datasets"][dataset] = record
        csv_rows.append({
            "dataset": dataset,
            "audit_pass": all_pass,
            "element_coverage_percent": coverage["element_coverage_percent"],
            "sample_any_hit_percent": coverage["sample_any_hit_percent"],
            "teacher_hit_mae": errors["teacher"]["exact_hit_mae_element_weighted"],
            "teacher_nonhit_mae": errors["teacher"]["non_hit_mae_element_weighted"],
            "network_hit_mae": errors["network_only"]["exact_hit_mae_element_weighted"],
            "network_nonhit_mae": errors["network_only"]["non_hit_mae_element_weighted"],
            "full_hit_mae": errors["full_system"]["exact_hit_mae_element_weighted"],
            "full_nonhit_mae": errors["full_system"]["non_hit_mae_element_weighted"],
        })
        print(f"{dataset}: PASS={all_pass}, exact coverage={coverage['element_coverage_percent']:.4f}%", flush=True)

    report["all_datasets_pass"] = bool(all(item["audit_pass"] for item in report["datasets"].values()))
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "coverage_and_grouped_errors.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    if not report["all_datasets_pass"]:
        raise SystemExit("at least one leakage audit failed")


if __name__ == "__main__":
    main()
