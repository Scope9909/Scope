"""Assemble the four-model PEMS04 efficiency table from frozen benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_results" / "icde_four_model_efficiency"

SOURCES = (
    ("ICASSP/Base Parent", "paper_results/hyperd_alignment_benchmark/base_parent_cuda_b64.json"),
    ("Joint3 Teacher", "paper_results/hyperd_alignment_benchmark_legacy/joint3_teacher_cuda_b64.json"),
    ("Lite-R256 network-only", "paper_results/hyperd_alignment_benchmark_legacy/lite_r256_kdfix3_s1_cuda_b64.json"),
    ("Lite-R256 + ValCal", "paper_results/hyperd_alignment_benchmark_legacy/lite_r256_kdfix3_valcal_s1_cuda_b64.json"),
)

OFFICIAL_MAE = {
    "ICASSP/Base Parent": 17.6639970025089,
    "Joint3 Teacher": 17.629655032101628,
    "Lite-R256 network-only": 17.6560046949234,
    "Lite-R256 + ValCal": 17.506396412911727,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    raw = {}
    reference = None
    for model, relative in SOURCES:
        path = ROOT / relative
        report = json.loads(path.read_text(encoding="utf-8"))
        raw[model] = report
        signature = {
            "gpu": report["device"]["gpu_name"],
            "torch": report["device"]["torch"],
            "cuda": report["device"].get("cuda_version", "11.7 (from torch build cu117)"),
            "python": report["device"]["python"],
            "split": report["split"],
            "batch_size": report["batch_size"],
            "warmup_batches": report["warmup_batches"],
            "repeats": report["repeats"],
            "logical_samples": report["logical_samples"],
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"benchmark protocol mismatch for {model}: {signature} != {reference}")
        files = {item["role"]: item for item in report["storage"]["files"]}
        calibration = report.get("calibration_benchmark")
        rows.append({
            "dataset": "PEMS04",
            "model": model,
            "batch_size": report["batch_size"],
            "repeats": report["repeats"],
            "parameters_total": report["model"]["parameters_total"],
            "parameters_trainable_flag_at_load": report["model"]["parameters_trainable_flag_at_load"],
            "checkpoint_mib": files.get("fp32_checkpoint_used", {}).get("mib", 0.0),
            "retrieval_cache_mib": files.get("retrieval_cache", {}).get("mib", 0.0),
            "candidate_cache_mib": files.get("candidate_cache", {}).get("mib", 0.0),
            "calibrator_mib": files.get("calibrator", {}).get("mib", 0.0),
            "complete_deployment_mib": report["storage"]["fp32_complete_deployment_mib"],
            "forward_ms_per_sample": report["timing"]["forward_ms_per_physical_sample"]["mean"],
            "end_to_end_ms_per_sample": report["timing"]["end_to_end_ms_per_physical_sample"]["mean"],
            "forward_throughput_samples_per_second": report["timing"]["forward_throughput_samples_per_second"],
            "peak_gpu_allocated_mib": report["memory"]["peak_gpu_allocated_mib"],
            "peak_gpu_reserved_mib": report["memory"]["peak_gpu_reserved_mib"],
            "valcal_cpu_ms_per_sample": (
                calibration["cpu_seconds_per_full_archive"]["mean_ms_per_sample"]
                if calibration else None
            ),
            "official_test_mae": OFFICIAL_MAE[model],
            "source_json": str(path.resolve()),
        })

    with (OUT / "four_model_efficiency.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": "pems04_four_model_same_hardware_batch64_v1",
        "environment": reference,
        "notes": [
            "All neural benchmarks use the same GPU, software stack, batch size, warm-up and repeats.",
            "ValCal is a frozen CPU post-processing step and its latency is reported separately.",
            "Complete deployment size includes checkpoint and all discovered caches/calibrator files.",
            "The ValCal row uses the frozen official calibrated Test MAE; its neural forward sanity MAE is the network-only MAE.",
        ],
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# ICDE four-model unified efficiency (PEMS04)", "",
        f"Environment: {reference['gpu']}; Torch {reference['torch']}; CUDA {reference['cuda']}; "
        f"batch={reference['batch_size']}; warm-up={reference['warmup_batches']}; repeats={reference['repeats']}.", "",
        "| Model | Params | Checkpoint MiB | Retrieval MiB | Candidate MiB | ValCal MiB | Deployment MiB | Forward ms/sample | E2E ms/sample | ValCal CPU ms/sample | Peak GPU MiB | Test MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cal = "-" if row["valcal_cpu_ms_per_sample"] is None else f"{row['valcal_cpu_ms_per_sample']:.4f}"
        lines.append(
            f"| {row['model']} | {row['parameters_total']:,} | {row['checkpoint_mib']:.2f} | "
            f"{row['retrieval_cache_mib']:.2f} | {row['candidate_cache_mib']:.2f} | "
            f"{row['calibrator_mib']:.2f} | {row['complete_deployment_mib']:.2f} | "
            f"{row['forward_ms_per_sample']:.4f} | {row['end_to_end_ms_per_sample']:.4f} | "
            f"{cal} | {row['peak_gpu_allocated_mib']:.2f} | {row['official_test_mae']:.4f} |"
        )
    lines += ["", "ValCal latency is measured separately on CPU and is not silently added to the GPU neural-forward column."]
    (OUT / "four_model_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
