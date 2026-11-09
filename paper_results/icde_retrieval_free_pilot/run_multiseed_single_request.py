"""Sequential Validation-only single-request replication for frozen PEMS models."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(__file__).with_name("benchmark_online_end_to_end.py")
PYTHON = ROOT / "_runtime_deps" / "python311_embed" / "python.exe"

CONFIGS = {
    "PEMS03": {
        "experiments": {
            1: "TLAST_improved_PEMS03_12to12_exp10101500_s1",
            2: "TLAST_improved_PEMS03_12to12_exp10101506_s2",
            3: "TLAST_improved_PEMS03_12to12_exp10101513_s3",
        },
        "calibration": "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}_valcal/calibration.json",
        "state_prefix": "pems03",
        "state_in_output_dir": True,
        "references": {
            "full": "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}_network_only/selected_predictions.npz",
        },
    },
    "PEMS04": {
        "experiment": "TLAST_improved_PEMS04_12to12_exppems04_joint3_lite_r256_strictalign_s{seed}",
        "calibration": "paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_s{seed}_valcal/calibration.json",
        "state_prefix": "pems04",
    },
    "PEMS07": {
        "experiment": "TLAST_improved_PEMS07_12to12_exppems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}",
        "calibration": "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}_valcal/calibration.json",
        "state_prefix": "pems07",
    },
    "PEMS08": {
        "experiment": "TLAST_improved_PEMS08_12to12_exppems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}",
        "calibration": "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}_valcal/calibration.json",
        "state_prefix": "pems08",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3])
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset in args.datasets:
        config = CONFIGS[dataset]
        for seed in args.seeds:
            exp_name = (config["experiments"][seed] if "experiments" in config else
                        config["experiment"].format(seed=seed))
            exp_dir = ROOT / "EXP_experiments" / "TLAST_improved" / dataset / exp_name
            calibration = ROOT / config["calibration"].format(seed=seed)
            if not (exp_dir / "best.pth").is_file() or not (exp_dir / "config.pkl").is_file():
                raise FileNotFoundError(f"Missing checkpoint/config: {exp_dir}")
            if not calibration.is_file():
                raise FileNotFoundError(calibration)
            for mode in ("backbone", "sparse", "full"):
                state = None
                if mode != "backbone":
                    state_root = (output_dir if config.get("state_in_output_dir") else
                                  ROOT / "cold_start_streaming")
                    state = state_root / f"{config['state_prefix']}_{mode}_cold_state.npz"
                    if not state.is_file():
                        raise FileNotFoundError(state)
                result = output_dir / f"{dataset.lower()}_s{seed}_{mode}_b1_r3.json"
                reference = None
                if mode in config.get("references", {}):
                    reference = ROOT / config["references"][mode].format(seed=seed)
                    if not reference.is_file():
                        raise FileNotFoundError(reference)
                jobs.append((dataset, seed, mode, exp_dir, calibration, state, reference, result))

    manifest = output_dir / ("run_manifest_" + "_".join(
        dataset.lower() for dataset in args.datasets) + ".json")
    for index, (dataset, seed, mode, exp_dir, calibration, state, reference, result) in enumerate(jobs, 1):
        if result.is_file():
            report = json.loads(result.read_text(encoding="utf-8"))
            if not (report["dataset"] == dataset and report["mode"] == mode and
                    report["batch_size"] == 1 and report["warmup_passes"] == 1 and
                    report["repeats"] == 3 and report["test_evaluated"] is False and
                    report["query_cache_loaded"] is False):
                raise ValueError(f"Existing result failed protocol check: {result}")
            print(f"[{index}/{len(jobs)}] existing {dataset} s{seed} {mode}", flush=True)
            continue
        command = [str(PYTHON), "-B", str(BENCHMARK),
                   "--exp-dir", str(exp_dir), "--mode", mode,
                   "--batch-size", "1", "--warmup-passes", "1", "--repeats", "3",
                   "--calibration", str(calibration), "--output", str(result)]
        if state is not None:
            command.extend(["--state", str(state)])
        if reference is not None:
            command.extend(["--reference-predictions", str(reference)])
        print(f"[{index}/{len(jobs)}] start {dataset} s{seed} {mode}", flush=True)
        log = output_dir / f"{dataset.lower()}_s{seed}_{mode}_b1_r3.log"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, check=False)
        if process.returncode != 0 or not result.is_file():
            raise RuntimeError(f"Benchmark failed; inspect {log}")
        report = json.loads(result.read_text(encoding="utf-8"))
        if not (report["test_evaluated"] is False and
                report["query_cache_loaded"] is False):
            raise ValueError(f"Invalid result protocol: {result}")
        print(f"[{index}/{len(jobs)}] done {dataset} s{seed} {mode}: "
              f"p50={report['end_to_end_batch_latency_ms_p50']:.3f} ms, "
              f"MAE={report['fixed_validation_calibration']['mean_fixed_calibrated_mae']:.6f}",
              flush=True)
        manifest.write_text(json.dumps({
            "protocol": "batch1_multiseed_validation_only_v1",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed": [job[-1].name for job in jobs if job[-1].is_file()],
            "planned": [job[-1].name for job in jobs],
            "test_evaluated": False,
        }, indent=2) + "\n", encoding="utf-8")
    print("All planned Validation jobs completed", flush=True)


if __name__ == "__main__":
    main()
