"""Summarize the authoritative Validation-only online-memory benchmarks."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "online_end_to_end"
FILES = {
    ("PEMS04", "backbone"): "pems04_backbone_e2e_r5.json",
    ("PEMS04", "sparse"): "pems04_sparse_e2e_r5_exactinput.json",
    ("PEMS04", "full"): "pems04_full_e2e_r5_exactinput.json",
    ("PEMS08", "backbone"): "pems08_backbone_e2e_r5.json",
    ("PEMS08", "sparse"): "pems08_sparse_e2e_r5.json",
    ("PEMS08", "full"): "pems08_full_e2e_r5.json",
    ("PEMS07", "backbone"): "pems07_backbone_e2e_r5.json",
    ("PEMS07", "sparse"): "pems07_sparse_e2e_r5_exactinput.json",
    ("PEMS07", "full"): "pems07_full_e2e_r5_exactinput.json",
}


def mib(value: int | float) -> float:
    return float(value) / (1024 ** 2)


def load_and_validate() -> dict[tuple[str, str], dict]:
    rows = {}
    for key, name in FILES.items():
        path = RESULT_DIR / name
        row = json.loads(path.read_text(encoding="utf-8"))
        dataset, mode = key
        if row["dataset"] != dataset or row["mode"] != mode:
            raise ValueError(f"Dataset/mode mismatch in {path}")
        if row["protocol"] != "train_only_online_memory_end_to_end_validation_v1":
            raise ValueError(f"Protocol mismatch in {path}")
        if row["test_evaluated"] or row["query_cache_loaded"]:
            raise ValueError(f"Closed-Test/cache invariant failed in {path}")
        if row["warmup_passes"] != 1 or row["repeats"] != 5 or row["batch_size"] != 32:
            raise ValueError(f"Timing protocol mismatch in {path}")
        parity = row.get("prediction_parity")
        if not parity or parity["max_absolute_difference"] != 0.0:
            raise ValueError(f"Prediction parity failed in {path}")
        checkpoint_bytes = Path(row["checkpoint"]).stat().st_size
        calibration_bytes = Path(
            row["fixed_validation_calibration"]["reference"]).stat().st_size
        row["result_file"] = name
        row["checkpoint_bytes"] = checkpoint_bytes
        row["calibration_bytes"] = calibration_bytes
        row["partial_deployment_bytes"] = (
            checkpoint_bytes + calibration_bytes + row["state_bytes"])
        rows[key] = row
    return rows


def main() -> None:
    rows = load_and_validate()
    comparisons = {}
    for dataset in ("PEMS04", "PEMS08", "PEMS07"):
        backbone = rows[(dataset, "backbone")]
        sparse = rows[(dataset, "sparse")]
        full = rows[(dataset, "full")]
        comparisons[dataset] = {
            "sparse_vs_full_pass_speedup": (
                full["pass_wall_median_seconds"] /
                sparse["pass_wall_median_seconds"]),
            "sparse_vs_full_batch_p50_speedup": (
                full["end_to_end_batch_latency_ms_p50"] /
                sparse["end_to_end_batch_latency_ms_p50"]),
            "full_over_sparse_state_ratio": full["state_bytes"] / sparse["state_bytes"],
            "sparse_state_reduction_percent": (
                100 * (1 - sparse["state_bytes"] / full["state_bytes"])),
            "sparse_vs_full_calibrated_mae_delta": (
                sparse["fixed_validation_calibration"]["mean_fixed_calibrated_mae"] -
                full["fixed_validation_calibration"]["mean_fixed_calibrated_mae"]),
            "sparse_vs_backbone_calibrated_mae_delta": (
                sparse["fixed_validation_calibration"]["mean_fixed_calibrated_mae"] -
                backbone["fixed_validation_calibration"]["mean_fixed_calibrated_mae"]),
        }

    serializable_rows = {
        f"{dataset}:{mode}": row for (dataset, mode), row in rows.items()
    }
    summary = {
        "protocol": "train_only_online_memory_end_to_end_validation_v1",
        "authoritative_result_files": list(FILES.values()),
        "test_evaluated": False,
        "query_cache_loaded": False,
        "batch_latency_scope": "one batch of at most 32 samples, not one request",
        "rows": serializable_rows,
        "comparisons": comparisons,
        "scope_note": (
            "PEMS04/08 are the primary cross-dataset evidence. PEMS07 is retained "
            "only as implementation-alignment and scale evidence because it informed "
            "the node-selection design."),
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Train-only online residual memory: Validation end-to-end benchmark",
        "",
        "**Protocol.** One full Validation pass, batch size 32, one warm-up pass, "
        "five measured passes. Retrieval is recomputed online from serialized "
        "Train-only state. No precomputed Validation query cache is loaded and Test "
        "is not evaluated. Every online prediction is bitwise identical to its "
        "corresponding cache-based reference (`max_abs_diff = 0`).",
        "",
        "Batch latency is for one batch of at most 32 samples, not a single request. "
        "Partial deployment size is checkpoint + fixed-calibration JSON + online "
        "state; it excludes the common runtime and code.",
        "",
        "| Dataset | Mode | Nodes | State MiB | Partial MiB | Pass median s | Pass p95 s | Batch p50 ms | Batch p95 ms | Throughput sample/s | Retrieval median s | CPU peak MiB | GPU alloc MiB | Raw Val MAE | Fixed-cal Val MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("PEMS04", "PEMS08", "PEMS07"):
        for mode in ("backbone", "sparse", "full"):
            row = rows[(dataset, mode)]
            lines.append(
                f"| {dataset} | {mode} | {row['selected_nodes']}/{row['total_nodes']} | "
                f"{mib(row['state_bytes']):.3f} | "
                f"{mib(row['partial_deployment_bytes']):.3f} | "
                f"{row['pass_wall_median_seconds']:.3f} | "
                f"{row['pass_wall_p95_seconds']:.3f} | "
                f"{row['end_to_end_batch_latency_ms_p50']:.2f} | "
                f"{row['end_to_end_batch_latency_ms_p95']:.2f} | "
                f"{row['throughput_samples_per_second_at_median']:.2f} | "
                f"{row['retrieval_median_seconds']:.3f} | "
                f"{mib(row['cpu_rss_peak_sampled_bytes']):.1f} | "
                f"{mib(row['gpu_peak_allocated_bytes']):.1f} | "
                f"{row['raw_validation_metrics']['mae']:.4f} | "
                f"{row['fixed_validation_calibration']['mean_fixed_calibrated_mae']:.4f} |")

    lines.extend(["", "## Sparse versus full", ""])
    for dataset in ("PEMS04", "PEMS08", "PEMS07"):
        item = comparisons[dataset]
        lines.append(
            f"- **{dataset}:** sparse is "
            f"{item['sparse_vs_full_pass_speedup']:.2f}x faster per full pass and "
            f"{item['sparse_vs_full_batch_p50_speedup']:.2f}x faster at batch p50; "
            f"state is reduced by {item['sparse_state_reduction_percent']:.2f}%; "
            f"fixed-calibrated MAE delta (sparse - full) is "
            f"{item['sparse_vs_full_calibrated_mae_delta']:+.4f}, and delta "
            f"(sparse - backbone) is "
            f"{item['sparse_vs_backbone_calibrated_mae_delta']:+.4f}.")

    lines.extend([
        "",
        "## Evidence boundary",
        "",
        "PEMS04 and PEMS08 are the primary cross-dataset evidence. PEMS07 is "
        "reported only as implementation-alignment and scale evidence because it "
        "informed the node-selection design; it is not claimed as an independent "
        "generalization dataset.",
        "",
        "The authoritative inputs are the nine files listed in `summary.json`. "
        "Earlier smoke checks and the superseded PEMS04 sparse run without exact "
        "integer input recovery are intentionally excluded.",
    ])
    (RESULT_DIR / "ONLINE_END_TO_END_VALIDATION_20261014.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
