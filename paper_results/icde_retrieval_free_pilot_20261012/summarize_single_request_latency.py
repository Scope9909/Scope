import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    records = {}
    for dataset in ("pems04", "pems08", "pems07"):
        records[dataset] = {}
        for mode in ("backbone", "sparse", "full"):
            path = args.input_dir / f"{dataset}_{mode}_b1_r3.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["dataset"].lower() == dataset
            assert data["mode"] == mode
            assert data["batch_size"] == 1
            assert data["warmup_passes"] == 1
            assert data["repeats"] == 3
            assert data["test_evaluated"] is False
            assert data["query_cache_loaded"] is False
            assert len(data["pass_wall_seconds"]) == 3
            parity = data["prediction_parity"]
            # Batch-size-dependent CUDA kernels are not bitwise invariant.
            assert parity["mean_absolute_difference"] < 1e-3
            assert parity["max_absolute_difference"] < 0.05
            records[dataset][mode] = data

    summary = {"protocol": "single_request_validation_batch1_v1", "datasets": {}}
    for dataset, modes in records.items():
        compact = {}
        for mode, data in modes.items():
            compact[mode] = {
                "state_bytes": data["state_bytes"],
                "selected_nodes": data["selected_nodes"],
                "pass_wall_seconds": data["pass_wall_seconds"],
                "pass_wall_median_seconds": data["pass_wall_median_seconds"],
                "throughput_samples_per_second": data["throughput_samples_per_second_at_median"],
                "request_p50_ms": data["end_to_end_batch_latency_ms_p50"],
                "request_p95_ms": data["end_to_end_batch_latency_ms_p95"],
                "retrieval_median_seconds": data["retrieval_median_seconds"],
                "cpu_peak_bytes": data["cpu_rss_peak_sampled_bytes"],
                "gpu_peak_allocated_bytes": data["gpu_peak_allocated_bytes"],
                "fixed_calibrated_mae": data["fixed_validation_calibration"]["mean_fixed_calibrated_mae"],
                "batch32_reference_max_abs_drift": data["prediction_parity"]["max_absolute_difference"],
                "batch32_reference_mean_abs_drift": data["prediction_parity"]["mean_absolute_difference"],
            }
        sparse, full, backbone = modes["sparse"], modes["full"], modes["backbone"]
        compact["sparse_vs_full"] = {
            "pass_speedup": full["pass_wall_median_seconds"] / sparse["pass_wall_median_seconds"],
            "request_p50_speedup": full["end_to_end_batch_latency_ms_p50"] / sparse["end_to_end_batch_latency_ms_p50"],
            "request_p95_speedup": full["end_to_end_batch_latency_ms_p95"] / sparse["end_to_end_batch_latency_ms_p95"],
            "state_reduction_percent": 100.0 * (1.0 - sparse["state_bytes"] / full["state_bytes"]),
            "fixed_mae_delta_sparse_minus_full": sparse["fixed_validation_calibration"]["mean_fixed_calibrated_mae"] - full["fixed_validation_calibration"]["mean_fixed_calibrated_mae"],
        }
        compact["sparse_vs_backbone"] = {
            "request_p50_overhead_ms": sparse["end_to_end_batch_latency_ms_p50"] - backbone["end_to_end_batch_latency_ms_p50"],
            "request_p95_overhead_ms": sparse["end_to_end_batch_latency_ms_p95"] - backbone["end_to_end_batch_latency_ms_p95"],
            "fixed_mae_delta_sparse_minus_backbone": sparse["fixed_validation_calibration"]["mean_fixed_calibrated_mae"] - backbone["fixed_validation_calibration"]["mean_fixed_calibrated_mae"],
        }
        summary["datasets"][dataset.upper()] = compact

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Validation-only single-request latency (batch size 1)", "",
        "Protocol: one complete warm-up pass followed by three complete Validation replays; batch size 1; no query cache; Test remains closed. Latencies include loader assembly, online causal retrieval, host-to-device transfer, model forward, and correction.", "",
        "| Dataset | Mode | pass median (s) | request p50 / p95 (ms) | throughput (req/s) | state bytes | fixed Val MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("PEMS04", "PEMS08", "PEMS07"):
        for mode in ("backbone", "sparse", "full"):
            row = summary["datasets"][dataset][mode]
            lines.append(f"| {dataset} | {mode} | {row['pass_wall_median_seconds']:.3f} | {row['request_p50_ms']:.3f} / {row['request_p95_ms']:.3f} | {row['throughput_samples_per_second']:.2f} | {row['state_bytes']:,} | {row['fixed_calibrated_mae']:.6f} |")
    lines += ["", "## Sparse versus full", "", "| Dataset | pass speedup | p50 speedup | p95 speedup | state reduction | MAE delta (sparse-full) |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("PEMS04", "PEMS08", "PEMS07"):
        row = summary["datasets"][dataset]["sparse_vs_full"]
        lines.append(f"| {dataset} | {row['pass_speedup']:.2f}x | {row['request_p50_speedup']:.2f}x | {row['request_p95_speedup']:.2f}x | {row['state_reduction_percent']:.2f}% | {row['fixed_mae_delta_sparse_minus_full']:+.6f} |")
    lines += [
        "", "## Interpretation", "",
        "These measurements represent sequential offline replay of individual Validation requests, not production concurrent-serving QPS. Sparse memory consistently reduces single-request latency and stored state relative to full memory. PEMS07 shows the largest latency gain because its full state spans all 883 sensors.", "",
        "Predictions are compared with existing batch-32 references. CUDA kernels can change floating-point accumulation with batch shape, so batch-1 results are not bitwise identical: all mean absolute drifts remain below 1e-3 and all maxima below 0.05. Exact retrieval parity is established separately by the batch-32 cold-start experiment.", "",
        "PEMS07 is protocol-aligned supporting evidence, but not an independent generalization dataset because it participated in selection-design work.", "",
    ]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
