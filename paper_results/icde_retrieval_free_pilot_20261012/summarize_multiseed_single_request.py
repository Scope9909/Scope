"""Summarize frozen three-seed, four-dataset single-request Validation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
            "per_seed": values.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed1-dir", type=Path, required=True)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    result = {"protocol": "frozen_multiseed_batch1_validation_only_v1",
              "test_evaluated": False, "datasets": {}}
    for dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        records = {}
        for seed in (1, 2, 3):
            records[seed] = {}
            for mode in ("backbone", "sparse", "full"):
                path = ((args.seed1_dir / f"{dataset.lower()}_{mode}_b1_r3.json")
                        if seed == 1 and dataset != "PEMS03" else
                        (args.multiseed_dir /
                         f"{dataset.lower()}_s{seed}_{mode}_b1_r3.json"))
                data = json.loads(path.read_text(encoding="utf-8"))
                if not (data["dataset"] == dataset and data["mode"] == mode and
                        data["batch_size"] == 1 and data["warmup_passes"] == 1 and
                        data["repeats"] == 3 and len(data["pass_wall_seconds"]) == 3 and
                        data["test_evaluated"] is False and
                        data["query_cache_loaded"] is False):
                    raise ValueError(f"Protocol mismatch: {path}")
                if mode != "backbone" and data["cold_start_endpoint"] is None:
                    raise ValueError(f"Missing cold-start state: {path}")
                records[seed][mode] = data
        dataset_summary = {"seeds": {}}
        for seed, modes in records.items():
            row = {}
            for mode, data in modes.items():
                row[mode] = {
                    "request_p50_ms": data["end_to_end_batch_latency_ms_p50"],
                    "request_p95_ms": data["end_to_end_batch_latency_ms_p95"],
                    "pass_median_seconds": data["pass_wall_median_seconds"],
                    "state_bytes": data["state_bytes"],
                    "selected_nodes": data["selected_nodes"],
                    "fixed_validation_mae": data["fixed_validation_calibration"]["mean_fixed_calibrated_mae"],
                    "raw_validation_mae": data["raw_validation_metrics"]["mae"],
                    "reference_drift": data["prediction_parity"],
                }
            row["sparse_minus_full_mae"] = (row["sparse"]["fixed_validation_mae"] -
                                            row["full"]["fixed_validation_mae"])
            row["sparse_minus_backbone_mae"] = (row["sparse"]["fixed_validation_mae"] -
                                                row["backbone"]["fixed_validation_mae"])
            row["sparse_vs_full_p50_speedup"] = (row["full"]["request_p50_ms"] /
                                                  row["sparse"]["request_p50_ms"])
            row["sparse_vs_full_p95_speedup"] = (row["full"]["request_p95_ms"] /
                                                  row["sparse"]["request_p95_ms"])
            row["sparse_vs_full_state_reduction_percent"] = 100 * (1 -
                row["sparse"]["state_bytes"] / row["full"]["state_bytes"])
            dataset_summary["seeds"][str(seed)] = row
        dataset_summary["aggregate"] = {}
        for mode in ("backbone", "sparse", "full"):
            dataset_summary["aggregate"][mode] = {
                field: stats([dataset_summary["seeds"][str(seed)][mode][field]
                              for seed in (1, 2, 3)])
                for field in ("request_p50_ms", "request_p95_ms",
                              "pass_median_seconds", "fixed_validation_mae")
            }
        for field in ("sparse_minus_full_mae", "sparse_minus_backbone_mae",
                      "sparse_vs_full_p50_speedup", "sparse_vs_full_p95_speedup",
                      "sparse_vs_full_state_reduction_percent"):
            dataset_summary["aggregate"][field] = stats([
                dataset_summary["seeds"][str(seed)][field] for seed in (1, 2, 3)])
        result["datasets"][dataset] = dataset_summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    lines = [
        "# Four-dataset, three-seed single-request Validation replication", "",
        "All runs: batch size 1, one full warm-up pass, three timed complete Validation passes, causal cold-start state, no query cache, no Test evaluation.", "",
        "| Dataset | Mode | p50 (ms), mean ± SD | p95 (ms), mean ± SD | Fixed Val MAE, mean ± SD |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, summary in result["datasets"].items():
        for mode in ("backbone", "sparse", "full"):
            row = summary["aggregate"][mode]
            format_stat = lambda x: f"{x['mean']:.4f} ± {x['sample_std']:.4f}"
            lines.append(f"| {dataset} | {mode} | {format_stat(row['request_p50_ms'])} | "
                         f"{format_stat(row['request_p95_ms'])} | "
                         f"{format_stat(row['fixed_validation_mae'])} |")
    lines += ["", "| Dataset | sparse/full p50 speedup, mean ± SD | sparse − full MAE, mean ± SD | state reduction |",
              "|---|---:|---:|---:|"]
    for dataset, summary in result["datasets"].items():
        agg = summary["aggregate"]
        speed = agg["sparse_vs_full_p50_speedup"]
        mae = agg["sparse_minus_full_mae"]
        reduction = agg["sparse_vs_full_state_reduction_percent"]["mean"]
        lines.append(f"| {dataset} | {speed['mean']:.3f} ± {speed['sample_std']:.3f}× | "
                     f"{mae['mean']:+.4f} ± {mae['sample_std']:.4f} | {reduction:.2f}% |")
    lines += [
        "", "PEMS03 uses its original float32 retrieval format; PEMS04/07/08 use Train-fitted INT8. Byte totals across these codecs are not directly matched.",
        "", "The PEMS03 node set is selected from the first 55% of seed-1 Validation by the frozen three-block stability rule and reused for seeds 2 and 3. PEMS07 previously participated in rule design. These results test seed stability and online implementation; neither dataset serves as untouched independent generalization evidence.",
        "", "Request latency is measured in sequential offline replay; it is not concurrent production QPS.", "",
    ]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("Validated and summarized 36 batch-1 Validation runs")


if __name__ == "__main__":
    main()
