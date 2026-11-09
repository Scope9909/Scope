"""Audit a compact Train-only residual-memory state on Validation queries.

Only Train raw frames/endpoints, selected IDs and frozen Train-fit scales are
serialized. Validation inputs are supplied at query time; no Validation query
result or label is stored in the deployable state. This is a retrieval audit,
not a timed end-to-end model serving benchmark.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analog_distance import masked_history_distance  # noqa: E402
from utils.datasets_utils import file_path_dict, load_raw_data  # noqa: E402


def packed_state(state: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **state)
    return buffer.getvalue()


def nonzero_mean(values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    valid = values != 0
    total = np.where(valid, values, 0.0).sum(axis=axis, dtype=np.float64)
    count = valid.sum(axis=axis)
    mean = np.zeros_like(total, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return mean, count


def online_values(state: dict[str, np.ndarray], query_history: np.ndarray,
                  query_tod: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_raw = state["train_raw"]
    idx_train = state["idx_train"]
    train_tod = state["train_tod"]
    input_length = int(state["input_length"])
    horizon = int(state["predict_length"])
    k = int(state["k"])
    min_valid = int(state["min_valid_history"])
    exact_min_valid = int(state["exact_min_valid_history"])
    nodes = len(state["node_ids"])
    x_offsets = np.arange(1 - input_length, 1, dtype=np.int64)
    y_offsets = np.arange(1, horizon + 1, dtype=np.int64)
    train_future = train_raw[idx_train[:, None] + y_offsets[None, :]]
    global_mean, global_count = nonzero_mean(train_future, axis=0)
    if np.any(global_count == 0):
        raise ValueError("Train has an empty fallback horizon/node")
    raw_retrieval = np.empty((len(query_history), horizon, nodes, 1), dtype=np.float32)
    raw_exact = np.zeros_like(raw_retrieval)
    availability = np.zeros(raw_retrieval.shape, dtype=np.uint8)
    for tod in np.unique(query_tod):
        positions = np.flatnonzero(query_tod == tod)
        endpoints = idx_train[train_tod == tod]
        if len(endpoints) < k:
            raise ValueError("Not enough Train candidates for a calendar bucket")
        candidate_history = train_raw[endpoints[:, None] + x_offsets[None, :]]
        q = query_history[positions]
        distance = masked_history_distance(q, candidate_history, min_valid,
                                           metric="masked_mse")
        nearest = np.argpartition(distance, kth=k - 1, axis=1)[:, :k, :]
        nearest_distance = np.take_along_axis(distance, nearest, axis=1)
        candidate_future = train_raw[endpoints[:, None] + y_offsets[None, :]].transpose(0, 2, 1)
        node_index = np.arange(nodes)[None, None, :]
        selected_future = candidate_future[nearest, node_index, :]
        selected_valid = np.isfinite(nearest_distance)[..., None] & (selected_future != 0)
        selected_count = selected_valid.sum(axis=1)
        selected_sum = np.where(selected_valid, selected_future, 0.0).sum(axis=1,
                                                                          dtype=np.float64)
        retrieved = np.zeros_like(selected_sum, dtype=np.float64)
        np.divide(selected_sum, selected_count, out=retrieved, where=selected_count > 0)
        tod_mean, tod_count = nonzero_mean(train_future[train_tod == tod], axis=0)
        fallback = np.where(tod_count > 0, tod_mean, global_mean).T
        retrieved = np.where(selected_count > 0, retrieved, fallback[None, :, :])
        raw_retrieval[positions, :, :, 0] = retrieved.transpose(0, 2, 1)

        query_expanded = q[:, None, :, :]
        candidate_expanded = candidate_history[None, :, :, :]
        valid = (query_expanded != 0) & (candidate_expanded != 0)
        diff = np.where(valid, query_expanded - candidate_expanded, 0.0)
        squared = np.einsum("qctn,qctn->qcn", diff, diff, optimize=True)
        exact = (valid.sum(axis=2) >= exact_min_valid) & (squared == 0)
        exact_valid = exact[..., None] & (candidate_future[None, :, :, :] != 0)
        exact_count = exact_valid.sum(axis=1)
        exact_total = np.where(exact_valid, candidate_future[None, :, :, :], 0.0).sum(
            axis=1, dtype=np.float64)
        exact_result = np.zeros_like(exact_total, dtype=np.float64)
        np.divide(exact_total, exact_count, out=exact_result, where=exact_count > 0)
        raw_exact[positions, :, :, 0] = exact_result.transpose(0, 2, 1)
        availability[positions, :, :, 0] = (exact_count > 0).transpose(
            0, 2, 1).astype(np.uint8)

    mean = state["scaler_mean"]
    std = state["scaler_std"]
    if mean.size == 1:
        mean = float(mean.reshape(-1)[0])
        std = float(std.reshape(-1)[0])
    else:
        mean = mean.reshape(1, 1, -1, 1)
        std = std.reshape(1, 1, -1, 1)
    retrieval = ((raw_retrieval - mean) / std).astype(np.float32)
    exact = ((raw_exact - mean) / std).astype(np.float32).astype(np.float16)
    quantized = np.clip(np.rint(retrieval / state["quant_scale"][None, ...]),
                        -127, 127).astype(np.int8)
    return quantized, exact, availability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("PEMS04", "PEMS07", "PEMS08"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--full-int8-cache", type=Path, required=True)
    parser.add_argument("--grouped-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--verify-existing-state", action="store_true",
                        help="Verify an allocator's existing Train-only state instead of writing it")
    parser.add_argument("--audit-full-compute", action="store_true",
                        help="Also time all-node exact-scan retrieval and verify it against the full cache")
    parser.add_argument("--benchmark-repeats", type=int, default=1,
                        help="Repeat retrieval-only batch computation for timing variability")
    args = parser.parse_args()
    if args.benchmark_repeats < 1:
        raise ValueError("--benchmark-repeats must be positive")
    if args.output_report.exists():
        raise FileExistsError("Refusing to overwrite report")
    if args.output_state.exists() != args.verify_existing_state:
        raise FileExistsError("State must exist exactly when --verify-existing-state is set")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    choices = np.asarray(plan["folds"][0]["heterogeneous_mode_by_node"])
    if np.any((choices != 0) & (choices != 1)):
        raise ValueError("Only OFF/INT8 plans supported")
    selected = np.flatnonzero(choices == 1).astype(np.int32)
    with np.load(args.full_int8_cache, allow_pickle=False) as full, np.load(
            args.grouped_cache, allow_pickle=False) as grouped:
        if not np.array_equal(selected, grouped["node_ids_int8"]):
            raise ValueError("Grouped cache node IDs differ from frozen plan")
        if not np.array_equal(full["idx_val"], grouped["idx_val"]):
            raise ValueError("Validation endpoints differ between caches")
        if (str(np.asarray(full["retrieval_quantization_fit_split"]).item()) != "train_only"
                or str(np.asarray(full["exact_match_rule"]).item()) != "masked_mse_eq_0"):
            raise ValueError("Unsupported retrieval/exact cache protocol")
        if ("calendar_mode" in full.files and str(full["calendar_mode"].item()) != "tod"):
            raise ValueError("Only same-time-of-day retrieval is supported")
        if ("aggregation" in full.files and str(full["aggregation"].item()) != "mean"):
            raise ValueError("Only mean retrieval is supported")
        idx_train = np.asarray(full["idx_train"], dtype=np.int64)
        idx_val = np.asarray(full["idx_val"], dtype=np.int64)
        length = int(full["input_length"])
        horizon = int(full["predict_length"])
        metadata = {key: np.asarray(full[key]) for key in
                    ("k", "min_valid_history", "input_length", "predict_length",
                     "exact_min_valid_history")}
        scale_all = np.asarray(full["retrieval_quantization_scale"])
        expected = {key: np.asarray(grouped[key]) for key in
                    ("retrieval_val_int8", "exact_retrieval_val_int8",
                     "exact_available_val_int8")}
        full_expected = ({key: np.asarray(full[key]) for key in
                          ("retrieval_val", "exact_retrieval_val", "exact_available_val")}
                         if args.audit_full_compute else None)
        mean_all = np.asarray(full["scaler_mean"])
        std_all = np.asarray(full["scaler_std"])
    data, _ = load_raw_data(file_path_dict[args.dataset])
    with np.load(ROOT / "datasets" / args.dataset / f"{args.dataset}_time.npz",
                 allow_pickle=False) as archive:
        time_data = np.asarray(archive["time"])
    if len(choices) != data.shape[1] or len(time_data) != len(data):
        raise ValueError("Plan/raw data/time dimensions differ")
    if (idx_train.min() < length - 1 or
            idx_train.max() + horizon >= len(data) or
            idx_val.min() < length - 1 or idx_val.max() >= len(data)):
        raise ValueError("Endpoint span outside available observations")
    last_train_frame = int(idx_train.max() + horizon)

    def state_for(ids: np.ndarray) -> dict[str, np.ndarray]:
        node_mean = mean_all if mean_all.size == 1 else mean_all[ids]
        node_std = std_all if std_all.size == 1 else std_all[ids]
        return {
            "train_raw": np.ascontiguousarray(data[:last_train_frame + 1, ids, 0],
                                                dtype=np.float32),
            "idx_train": idx_train.astype(np.int32),
            "train_tod": np.asarray(time_data[idx_train, 0], dtype=np.int16),
            "node_ids": ids.astype(np.int32),
            "quant_scale": np.ascontiguousarray(scale_all[:, ids, :]),
            "scaler_mean": np.asarray(node_mean),
            "scaler_std": np.asarray(node_std),
            **metadata,
        }

    state = state_for(selected)
    serialized_state = packed_state(state)
    if args.verify_existing_state:
        if args.output_state.read_bytes() != serialized_state:
            raise AssertionError("Existing Train-only state differs from the frozen plan")
    else:
        args.output_state.parent.mkdir(parents=True, exist_ok=True)
        args.output_state.write_bytes(serialized_state)
    full_state_bytes = len(packed_state(state_for(np.arange(len(choices), dtype=np.int32))))
    with np.load(args.output_state, allow_pickle=False) as saved:
        loaded = {key: saved[key] for key in saved.files}
    x_offsets = np.arange(1 - length, 1, dtype=np.int64)
    query_history = np.asarray(data[idx_val[:, None] + x_offsets[None, :]][:, :, selected, 0],
                               dtype=np.float32)
    query_tod = np.asarray(time_data[idx_val, 0], dtype=np.int16)
    for tod in np.unique(query_tod):
        candidate_endpoints = idx_train[loaded["train_tod"] == tod]
        query_endpoints = idx_val[query_tod == tod]
        if len(candidate_endpoints) and candidate_endpoints.max() + horizon > query_endpoints.min():
            raise ValueError("A same-TOD Train candidate future is not yet observable at a Validation query")
    selected_times = []
    for _ in range(args.benchmark_repeats):
        start = time.perf_counter()
        retrieval, exact, available = online_values(loaded, query_history, query_tod)
        selected_times.append(time.perf_counter() - start)
    online_seconds = selected_times[0]
    exact_ref = expected["exact_retrieval_val_int8"]
    availability_ref = expected["exact_available_val_int8"]
    retrieval_ref = expected["retrieval_val_int8"]
    mismatch_counts = (int(np.count_nonzero(retrieval != retrieval_ref)),
                       int(np.count_nonzero(exact != exact_ref)),
                       int(np.count_nonzero(available != availability_ref)))
    if any(mismatch_counts):
        raise AssertionError(f"Online retrieval parity failed: {mismatch_counts}")
    report = {
        "protocol": "train_only_online_retrieval_validation_parity_audit",
        "dataset": args.dataset,
        "test_evaluated": False,
        "selected_nodes": len(selected),
        "validation_queries": len(idx_val),
        "online_state_bytes": args.output_state.stat().st_size,
        "full_node_online_state_bytes_in_memory": full_state_bytes,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "calibration_bytes": args.calibration.stat().st_size,
        "partial_selected_deployment_bytes": (args.output_state.stat().st_size +
                                              args.checkpoint.stat().st_size +
                                              args.calibration.stat().st_size),
        "partial_full_deployment_bytes": (full_state_bytes + args.checkpoint.stat().st_size +
                                          args.calibration.stat().st_size),
        "query_cache_in_deployment": False,
        "index_definition": "Train endpoint and time-of-day arrays; exact scan, no approximate index",
        "online_retrieval_seconds_for_all_validation_queries": online_seconds,
        "selected_batch_retrieval_seconds_repeats": selected_times,
        "selected_batch_retrieval_seconds_median": float(np.median(selected_times)),
        "retrieval_int8_mismatch_count": mismatch_counts[0],
        "exact_fp16_mismatch_count": mismatch_counts[1],
        "availability_mismatch_count": mismatch_counts[2],
        "exact_fp16_max_absolute_difference": float(np.max(np.abs(
            exact.astype(np.float32) - exact_ref.astype(np.float32)))),
        "note": "Query histories supplied from observable Validation inputs; no Validation labels or query outputs stored. Timing is retrieval only, not model serving.",
    }
    if args.audit_full_compute:
        all_ids = np.arange(len(choices), dtype=np.int32)
        full_query_history = np.asarray(
            data[idx_val[:, None] + x_offsets[None, :]][:, :, all_ids, 0],
            dtype=np.float32)
        full_state = state_for(all_ids)
        full_times = []
        for _ in range(args.benchmark_repeats):
            tic = time.perf_counter()
            full_retrieval, full_exact, full_available = online_values(
                full_state, full_query_history, query_tod)
            full_times.append(time.perf_counter() - tic)
        full_seconds = full_times[0]
        full_mismatch = (
            int(np.count_nonzero(full_retrieval != full_expected["retrieval_val"])),
            int(np.count_nonzero(full_exact != full_expected["exact_retrieval_val"])),
            int(np.count_nonzero(full_available != full_expected["exact_available_val"])),
        )
        if any(full_mismatch):
            raise AssertionError(f"Full-node online retrieval parity failed: {full_mismatch}")
        report["full_node_online_retrieval_seconds_for_all_validation_queries"] = full_seconds
        report["full_batch_retrieval_seconds_repeats"] = full_times
        report["full_batch_retrieval_seconds_median"] = float(np.median(full_times))
        report["full_node_retrieval_mismatch_counts"] = full_mismatch
        report["selected_vs_full_retrieval_time_ratio_single_run"] = online_seconds / full_seconds
        report["selected_vs_full_retrieval_time_ratio_median"] = (
            float(np.median(selected_times) / np.median(full_times)))
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
