"""Build auditable ICDE tables from completed experiment artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def log_stats(directory: str) -> dict:
    paths = list((ROOT / directory).glob("*.log"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one log under {directory}, found {paths}")
    text = paths[0].read_text(encoding="utf-8", errors="ignore")
    anchor_match = re.search(r"Epoch 000 anchor,.*?Valid MAE: ([0-9.]+)", text)
    parameter_match = re.search(r"total_parameter_num is (\d+)", text)
    rows = [
        (int(epoch), float(mae), float(rmse))
        for epoch, mae, rmse in re.findall(
            r"Epoch: (\d+),.*?Valid MAE: ([0-9.]+).*?Valid RMSE: ([0-9.]+)",
            text,
        )
    ]
    if not rows:
        raise RuntimeError(f"no validation epochs in {paths[0]}")
    best = min(rows, key=lambda row: row[1])
    config = json.loads((ROOT / directory / "config.json").read_text(encoding="utf-8"))
    return {
        "rank": int(config["predictor_factor_rank"]),
        "parameters": int(parameter_match.group(1)) if parameter_match else None,
        "anchor_val_mae": float(anchor_match.group(1)) if anchor_match else None,
        "best_epoch": best[0],
        "best_val_mae": best[1],
        "best_val_rmse": best[2],
        "training_finished": "Training finished" in text,
        "test_skipped": "Test evaluation intentionally skipped" in text,
    }


def weighted_fold_metrics(report: dict, stage: str) -> dict:
    folds = report["validation_folds"]
    weights = [fold["evaluation_end"] - fold["evaluation_start"] for fold in folds]
    total = sum(weights)
    return {
        key: sum(fold[stage][key] * weight for fold, weight in zip(folds, weights)) / total
        for key in ("mae", "rmse", "mape")
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out = ROOT / "paper_results" / "icde_formal_tables_20261011"
    out.mkdir(parents=True, exist_ok=True)

    calibrations = {
        "PEMS04": load_json("paper_results/lite_r256_kdfix3_a30_lr1e4_smooth_tod_s1/calibration.json"),
        "PEMS07": load_json("paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/calibration.json"),
        "PEMSBAY": load_json("paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/calibration.json"),
    }
    ablation_rows = []
    for dataset, report in calibrations.items():
        for stage, key in (
            ("Lite-R256", "baseline_test"),
            ("+ smooth-level", "smooth_level_test"),
            ("+ target-time (ValCal)", "calibrated_test"),
        ):
            values = report[key]
            ablation_rows.append({
                "dataset": dataset, "stage": stage, "scope": "Test (frozen)",
                "mae": values["mae"], "rmse": values["rmse"],
                "mape_percent": 100 * values["mape"],
            })
        for stage, key in (
            ("Lite-R256", "baseline"),
            ("+ smooth-level", "smooth_level"),
            ("+ target-time (ValCal)", "smooth_plus_target_time"),
        ):
            values = weighted_fold_metrics(report, key)
            ablation_rows.append({
                "dataset": dataset, "stage": stage,
                "scope": "Validation expanding-fold OOS",
                "mae": values["mae"], "rmse": values["rmse"],
                "mape_percent": 100 * values["mape"],
            })
    write_csv(out / "formal_ablation.csv", ablation_rows)

    rank_dirs = {
        128: "EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_explite_r128_kdfix3_a30_lr1e4_val_s1_strict_20261011",
        256: "EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_explite_r256_kdfix3_a30_lr1e4_val_s1",
        384: "EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_explite_r384_kdfix3_a30_lr1e4_val_s1_strict_20261011",
    }
    rank_rows = [log_stats(rank_dirs[rank]) for rank in (128, 256, 384)]
    write_csv(out / "rank_sensitivity_validation.csv", rank_rows)

    kd_rows = []
    for alpha, name in (
        (0.1, "lite_r256_kdfix3_a10_lr1e4_val_s1"),
        (0.2, "lite_r256_kdfix3_a20_lr1e4_val_s1"),
        (0.3, "lite_r256_kdfix3_a30_lr1e4_val_s1"),
    ):
        stats = log_stats(
            f"EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_exp{name}"
        )
        kd_rows.append({"kd_alpha": alpha, **stats})
    write_csv(out / "kd_alpha_sensitivity_validation.csv", kd_rows)

    efficiency_sources = (
        ("PEMS04", "Teacher", "paper_results/unified_efficiency_b1_b32_20261011/pems04_teacher_cuda_b1.json"),
        ("PEMS04", "Teacher", "paper_results/unified_efficiency_b1_b32_20261011/pems04_teacher_cuda_b32.json"),
        ("PEMS04", "Lite-R256+ValCal", "paper_results/unified_efficiency_b1_b32_20261011/pems04_lite_r256_valcal_cuda_b1.json"),
        ("PEMS04", "Lite-R256+ValCal", "paper_results/unified_efficiency_b1_b32_20261011/pems04_lite_r256_valcal_cuda_b32.json"),
        ("PEMS07", "Teacher", "paper_results/unified_efficiency_b1_b32_20261011/pems07_teacher_cuda_b1.json"),
        ("PEMS07", "Teacher", "paper_results/unified_efficiency_b1_b32_20261011/pems07_teacher_cuda_b32.json"),
        ("PEMS07", "Lite-R256+ValCal", "paper_results/unified_efficiency_b1_b32_20261011/pems07_lite_r256_valcal_cuda_b1.json"),
        ("PEMS07", "Lite-R256+ValCal", "paper_results/unified_efficiency_b1_b32_20261011/pems07_lite_r256_valcal_cuda_b32.json"),
    )
    efficiency_rows = []
    for dataset, model, source in efficiency_sources:
        report = load_json(source)
        calibration = report.get("calibration_benchmark")
        efficiency_rows.append({
            "dataset": dataset, "model": model, "batch_size": report["batch_size"],
            "passes": report["repeats"],
            "forward_ms_per_sample": report["timing"]["forward_ms_per_physical_sample"]["mean"],
            "end_to_end_ms_per_sample": report["timing"]["end_to_end_ms_per_physical_sample"]["mean"],
            "throughput_samples_per_second": report["timing"]["forward_throughput_samples_per_second"],
            "calibrator_cpu_ms_per_sample": (
                calibration["cpu_seconds_per_full_archive"]["mean_ms_per_sample"]
                if calibration else None
            ),
            "parameters": report["model"]["parameters_total"],
            "deployment_mib": report["storage"]["fp32_complete_deployment_mib"],
            "peak_gpu_allocated_mib": report["memory"]["peak_gpu_allocated_mib"],
        })
    write_csv(out / "unified_efficiency.csv", efficiency_rows)

    horizon = load_json("paper_results/icde_core_horizon_bootstrap_20261011/summary.json")
    quantization = load_json("paper_results/pems07_valcal_quantization_20261011/quantization_report.json")
    summary = {
        "ablation": ablation_rows,
        "rank_sensitivity": rank_rows,
        "kd_alpha_sensitivity": kd_rows,
        "efficiency": efficiency_rows,
        "paired_bootstrap": {
            dataset: values["paired_bootstrap_seed1"]
            for dataset, values in horizon["datasets"].items()
        },
        "pems07_valcal_quantization": quantization,
    }
    (out / "all_tables.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# ICDE formal experiment tables (2026-10-11)", "",
        "Protocol: all choices are Validation-only; rank sensitivity never opens Test.", "",
        "## Formal calibration ablation", "",
        "| Dataset | Stage | Scope | MAE | RMSE | MAPE |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in ablation_rows:
        lines.append(
            f"| {row['dataset']} | {row['stage']} | {row['scope']} | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['mape_percent']:.4f}% |"
        )
    lines += ["", "## PEMS04 strict rank sensitivity (Validation only)", "",
              "| Rank | Anchor MAE | Best epoch | Best Val MAE | Best Val RMSE | Test skipped |",
              "|---:|---:|---:|---:|---:|---|" ]
    for row in rank_rows:
        lines.append(
            f"| {row['rank']} | {row['anchor_val_mae']:.4f} | {row['best_epoch']} | "
            f"{row['best_val_mae']:.4f} | {row['best_val_rmse']:.4f} | {row['test_skipped']} |"
        )
    lines += ["", "## Unified efficiency", "",
              "| Dataset | Model | Batch | Forward ms/sample | E2E ms/sample | Calibrator CPU ms/sample | Params | Deployment MiB |",
              "|---|---|---:|---:|---:|---:|---:|---:|" ]
    for row in efficiency_rows:
        cal = "—" if row["calibrator_cpu_ms_per_sample"] is None else f"{row['calibrator_cpu_ms_per_sample']:.4f}"
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['batch_size']} | "
            f"{row['forward_ms_per_sample']:.4f} | {row['end_to_end_ms_per_sample']:.4f} | "
            f"{cal} | {row['parameters']:,} | {row['deployment_mib']:.2f} |"
        )
    (out / "ICDE_formal_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
