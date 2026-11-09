"""Resumable fixed-teacher Lite-R256 -> Validation calibration pipeline.

For each dataset/seed this driver enforces the paper protocol:
1. train with ``validation_only=true`` and select ``best.pth`` by Validation;
2. run the selected checkpoint once to export Validation/Test predictions;
3. fit the fixed calibrator on Validation, freeze it, then report Test once.

Existing completed stages are skipped, making the command safe to resume.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_TEMPLATE = (
    "cross_dataset/{dataset}_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1.json"
)


def run(command: list[str], log_path: Path | None = None) -> None:
    print("[RUN]", subprocess.list2cmdline(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def training_complete(exp_dir: Path) -> bool:
    if not (exp_dir / "best.pth").is_file():
        return False
    for path in exp_dir.glob("*.log"):
        if "Training finished" in path.read_text(encoding="utf-8", errors="ignore"):
            return True
    return False


def matching_completed_exp(dataset: str, seed: int, expected: Path) -> Path | None:
    """Find an older timestamp-named run without guessing from its directory name."""
    if training_complete(expected):
        return expected
    parent = ROOT / "EXP_experiments" / "TLAST_improved" / dataset
    matches = []
    for candidate in parent.glob(f"TLAST_improved_{dataset}_12to12_exp*"):
        config_path = candidate / "config.pkl"
        if not config_path.is_file() or not training_complete(candidate):
            continue
        try:
            with config_path.open("rb") as stream:
                saved = pickle.load(stream)
        except Exception:
            continue
        identity = (
            getattr(saved, "dataset_name", None) == dataset
            and int(getattr(saved, "seed", -1)) == seed
            and getattr(saved, "predictor_type", None) == "richer_factorized"
            and int(getattr(saved, "predictor_factor_rank", -1)) == 256
            and abs(float(getattr(saved, "kd_alpha", -1.0)) - 0.3) < 1e-12
            and bool(getattr(saved, "validation_only", False))
            and str(getattr(saved, "frozen_parent_ckpt", "")).endswith(
                f"{dataset}_12to12_exppaper_{dataset.lower()}_s1_exact_k4/best.pth"
            )
        )
        if identity:
            matches.append(candidate)
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple completed runs match {dataset} seed={seed}: {matches}"
        )
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["PEMS03", "PEMS07"],
        choices=("PEMS03", "PEMS07", "PEMS08"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        dataset_lower = dataset.lower()
        config = CONFIG_TEMPLATE.format(dataset=dataset_lower)
        config_path = ROOT / "configurations" / config
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        if not config_data["training_config"].get("validation_only", False):
            raise RuntimeError(f"{config_path} must set validation_only=true")

        for seed in args.seeds:
            run_id = f"{dataset_lower}_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}"
            expected_exp_dir = (
                ROOT / "EXP_experiments" / "TLAST_improved" / dataset
                / f"TLAST_improved_{dataset}_12to12_exp{run_id}"
            )
            exp_dir = matching_completed_exp(
                dataset, seed, expected_exp_dir
            ) or expected_exp_dir
            prediction_dir = ROOT / "paper_results" / f"{run_id}_network_only"
            prediction_path = prediction_dir / "selected_predictions.npz"
            calibration_dir = ROOT / "paper_results" / f"{run_id}_valcal"

            commands = []
            if not training_complete(exp_dir):
                commands.append(([
                    sys.executable, "run.py", "--config_file", config,
                    "--seed", str(seed), "--run_id", run_id,
                ], ROOT / f"_{dataset_lower}_lite_r256_s{seed}_console.log"))
            if not prediction_path.is_file():
                commands.append(([
                    sys.executable, "export_checkpoint_predictions_for_calibration.py",
                    "--exp-dir", str(exp_dir), "--output", str(prediction_path),
                ], ROOT / f"_{dataset_lower}_lite_r256_s{seed}_export.log"))
            calibration_json = calibration_dir / "calibration.json"
            if not calibration_json.is_file():
                if calibration_dir.exists():
                    raise RuntimeError(
                        f"partial calibration directory exists; inspect it first: {calibration_dir}"
                    )
                commands.append(([
                    sys.executable, "calibrate_dualscale_smooth_tod.py",
                    "--predictions", str(prediction_path),
                    "--output-dir", str(calibration_dir),
                    "--num-bins", "20", "--smooth-shrinkage", "0.55",
                    "--num-time-bins", "96", "--time-shrinkage", "0.35",
                    "--slices-per-day", "288",
                ], ROOT / f"_{dataset_lower}_lite_r256_s{seed}_valcal.log"))

            if not commands:
                print(f"[SKIP] {dataset} seed={seed}: complete", flush=True)
            for command, log_path in commands:
                if args.dry_run:
                    print("[DRY-RUN]", subprocess.list2cmdline(command))
                else:
                    run(command, log_path)


if __name__ == "__main__":
    main()
