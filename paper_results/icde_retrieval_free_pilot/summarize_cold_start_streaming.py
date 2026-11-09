"""Validate and summarize causal cold-start streaming Validation evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cold_start_streaming"
RESULTS = {
    ("PEMS04", "sparse"): "pems04_sparse_incremental_r5.json",
    ("PEMS04", "full"): "pems04_full_incremental_r5.json",
    ("PEMS08", "sparse"): "pems08_sparse_incremental_r5.json",
    ("PEMS08", "full"): "pems08_full_incremental_r5.json",
    ("PEMS07", "sparse"): "pems07_sparse_incremental_r5.json",
    ("PEMS07", "full"): "pems07_full_incremental_r5.json",
}
STATES = {
    (dataset, mode): f"{dataset.lower()}_{mode}_cold_state.npz"
    for dataset, mode in RESULTS
}
STATE_REPORTS = {
    key: name.replace(".npz", ".json") for key, name in STATES.items()
}
ORIGINAL = {
    ("PEMS04", "sparse"): ROOT / "online_end_to_end" /
        "pems04_sparse_e2e_r5_exactinput.json",
    ("PEMS04", "full"): ROOT / "online_end_to_end" /
        "pems04_full_e2e_r5_exactinput.json",
    ("PEMS08", "sparse"): ROOT / "online_end_to_end" /
        "pems08_sparse_e2e_r5.json",
    ("PEMS08", "full"): ROOT / "online_end_to_end" /
        "pems08_full_e2e_r5.json",
    ("PEMS07", "sparse"): ROOT / "online_end_to_end" /
        "pems07_sparse_e2e_r5_exactinput.json",
    ("PEMS07", "full"): ROOT / "online_end_to_end" /
        "pems07_full_e2e_r5_exactinput.json",
}


def mib(value: int | float) -> float:
    return float(value) / (1024 ** 2)


def main() -> None:
    rows, state_rows, comparisons = {}, {}, {}
    forbidden = {"val_prediction", "val_truth", "val_endpoints",
                 "test_prediction", "test_truth", "test_endpoints",
                 "query_output", "labels"}
    for key, filename in RESULTS.items():
        dataset, mode = key
        result = json.loads((OUT / filename).read_text(encoding="utf-8"))
        state_report = json.loads(
            (OUT / STATE_REPORTS[key]).read_text(encoding="utf-8"))
        original = json.loads(ORIGINAL[key].read_text(encoding="utf-8"))
        if result["dataset"] != dataset or result["mode"] != mode:
            raise ValueError(f"Result identity mismatch: {filename}")
        if (result["test_evaluated"] or result["query_cache_loaded"] or
                result["warmup_passes"] != 1 or result["repeats"] != 5 or
                result["batch_size"] != 32 or
                result["pending_candidates_at_cold_start"] != 11 or
                result["prediction_parity"]["max_absolute_difference"] != 0.0 or
                result["prediction_parity"]["mean_absolute_difference"] != 0.0):
            raise ValueError(f"Result invariant failed: {filename}")
        if (result["raw_validation_metrics"] != original["raw_validation_metrics"] or
                result["fixed_validation_calibration"][
                    "mean_fixed_calibrated_mae"] != original[
                        "fixed_validation_calibration"][
                            "mean_fixed_calibrated_mae"]):
            raise ValueError(f"Embargo changed accuracy: {filename}")
        state_path = OUT / STATES[key]
        with np.load(state_path, allow_pickle=False) as archive:
            if forbidden.intersection(archive.files):
                raise ValueError(f"Serialized forbidden outputs: {state_path}")
            cold = int(archive["cold_start_endpoint"])
            horizon = int(archive["predict_length"])
            active = np.asarray(archive["idx_train"])
            pending = np.asarray(archive["pending_idx_train"])
            if (len(archive["train_raw"]) - 1 != cold or
                    np.max(active) + horizon > cold or
                    len(pending) != 11 or np.min(pending) + horizon <= cold):
                raise ValueError(f"State causality failed: {state_path}")
        if (state_report["unobservable_future_frames_removed"] != 11 or
                state_report["raw_pending_futures_serialized"] or
                state_report["query_outputs_serialized"] or
                state_report["validation_labels_serialized"] or
                state_report["test_evaluated"]):
            raise ValueError(f"State report invariant failed: {state_path}")
        result["result_file"] = filename
        result["state_report_file"] = STATE_REPORTS[key]
        result["pass_wall_range_seconds"] = [
            min(result["pass_wall_seconds"]), max(result["pass_wall_seconds"])]
        result["relative_to_non_embargo_pass_percent"] = 100 * (
            result["pass_wall_median_seconds"] /
            original["pass_wall_median_seconds"] - 1)
        rows[f"{dataset}:{mode}"] = result
        state_rows[f"{dataset}:{mode}"] = state_report

    for dataset in ("PEMS04", "PEMS08", "PEMS07"):
        sparse = rows[f"{dataset}:sparse"]
        full = rows[f"{dataset}:full"]
        comparisons[dataset] = {
            "sparse_vs_full_pass_speedup": (
                full["pass_wall_median_seconds"] /
                sparse["pass_wall_median_seconds"]),
            "sparse_vs_full_batch_p50_speedup": (
                full["end_to_end_batch_latency_ms_p50"] /
                sparse["end_to_end_batch_latency_ms_p50"]),
            "sparse_state_reduction_percent": 100 * (
                1 - sparse["state_bytes"] / full["state_bytes"]),
            "sparse_minus_full_fixed_calibrated_mae": (
                sparse["fixed_validation_calibration"][
                    "mean_fixed_calibrated_mae"] -
                full["fixed_validation_calibration"][
                    "mean_fixed_calibrated_mae"]),
        }

    summary = {
        "protocol": "causal_cold_start_streaming_validation_v1",
        "test_evaluated": False,
        "query_cache_loaded": False,
        "authoritative_result_files": list(RESULTS.values()),
        "cold_start_policy": (
            "Snapshot includes observations through the first Validation query "
            "endpoint. Eleven Train candidates with incomplete futures serialize "
            "only endpoint/TOD metadata and activate from the live query stream."),
        "rows": rows,
        "state_reports": state_rows,
        "comparisons": comparisons,
        "scope_note": (
            "PEMS04/08 are primary cross-dataset evidence; PEMS07 is scale and "
            "implementation-alignment evidence, not independent generalization."),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Causal cold-start streaming memory — Validation-only evidence",
        "",
        "The initial state contains observations only through the first Validation "
        "query endpoint. The 11 boundary Train candidates whose futures are not "
        "yet complete store endpoint/TOD metadata only and are activated from live "
        "query observations. No future values, Validation labels, query outputs, "
        "or Test data are serialized.",
        "",
        "Protocol: batch size 32, one warm-up pass, five measured complete "
        "Validation passes. All six prediction arrays are exactly equal to the "
        "cache-based references (`max_abs_diff = mean_abs_diff = 0`).",
        "",
        "| Dataset | Mode | Nodes | State MiB | Pass median s | Range s | Batch p50/p95 ms | Retrieval median s | CPU peak MiB | GPU alloc MiB | Fixed-cal Val MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("PEMS04", "PEMS08", "PEMS07"):
        for mode in ("sparse", "full"):
            row = rows[f"{dataset}:{mode}"]
            low, high = row["pass_wall_range_seconds"]
            lines.append(
                f"| {dataset} | {mode} | {row['selected_nodes']}/{row['total_nodes']} | "
                f"{mib(row['state_bytes']):.3f} | "
                f"{row['pass_wall_median_seconds']:.3f} | {low:.3f}–{high:.3f} | "
                f"{row['end_to_end_batch_latency_ms_p50']:.2f}/"
                f"{row['end_to_end_batch_latency_ms_p95']:.2f} | "
                f"{row['retrieval_median_seconds']:.3f} | "
                f"{mib(row['cpu_rss_peak_sampled_bytes']):.1f} | "
                f"{mib(row['gpu_peak_allocated_bytes']):.1f} | "
                f"{row['fixed_validation_calibration']['mean_fixed_calibrated_mae']:.4f} |")
    lines.extend(["", "## Sparse versus full", ""])
    for dataset, item in comparisons.items():
        lines.append(
            f"- **{dataset}:** {item['sparse_vs_full_pass_speedup']:.2f}x pass "
            f"speedup, {item['sparse_vs_full_batch_p50_speedup']:.2f}x batch-p50 "
            f"speedup, {item['sparse_state_reduction_percent']:.2f}% state "
            f"reduction, MAE delta (sparse - full) "
            f"{item['sparse_minus_full_fixed_calibrated_mae']:+.4f}.")
    lines.extend([
        "",
        "PEMS04/08 are the primary cross-dataset evidence. PEMS07 is retained only "
        "as scale and implementation-alignment evidence because it informed the "
        "node-selection design. Diagnostic and pre-optimization files are excluded; "
        "the six authoritative filenames are listed in `summary.json`.",
    ])
    (OUT / "COLD_START_STREAMING_VALIDATION_20261014.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
