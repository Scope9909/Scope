"""Run the pre-registered Lite-R256 validation-only KD sweep.

This helper never evaluates the test split.  It materializes one config per
setting under configurations/sweeps and launches run.py sequentially with
unique run IDs so existing experiments cannot be overwritten.
"""

from __future__ import annotations

import copy
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
BASE_CONFIG = ROOT / "configurations" / "PEMS04_lite_r256_activation_aware_distill.json"
CONFIG_DIR = ROOT / "configurations" / "sweeps"
EXP_ROOT = ROOT / "EXP_experiments" / "TLAST_improved" / "PEMS04"

# The original alpha=0.15, lr=2e-4 run is the fixed reference (Val 17.8431).
SWEEP = (
    (0.10, 2e-4),
    (0.20, 2e-4),
    (0.30, 2e-4),
    (0.10, 1e-4),
    (0.20, 1e-4),
    (0.30, 1e-4),
)


def tag(alpha: float, learning_rate: float) -> str:
    alpha_tag = f"{int(round(alpha * 100)):02d}"
    lr_tag = "2e4" if learning_rate == 2e-4 else "1e4"
    return f"lite_r256_kdfix3_a{alpha_tag}_lr{lr_tag}_val_s1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, default=BASE_CONFIG,
                        help='Base JSON path relative to the repository, or absolute.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show six planned runs without writing or training.')
    cli = parser.parse_args()
    base_path = cli.base_config if cli.base_config.is_absolute() else ROOT / cli.base_config
    if not PYTHON.exists():
        raise FileNotFoundError(f"Requested Python does not exist: {PYTHON}")
    with base_path.open("r", encoding="utf-8") as handle:
        base = json.load(handle)

    if not cli.dry_run:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for alpha, learning_rate in SWEEP:
        run_id = tag(alpha, learning_rate)
        exp_dir = EXP_ROOT / f"TLAST_improved_PEMS04_12to12_exp{run_id}"
        if cli.dry_run:
            print(f'[DRY-RUN] {run_id}: alpha={alpha}, lr={learning_rate}, exists={exp_dir.exists()}')
            continue
        if exp_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing experiment: {exp_dir}"
            )

        config = copy.deepcopy(base)
        config["model_config"]["kd_alpha"] = alpha
        config["training_config"]["learning_rate"] = learning_rate
        config["training_config"]["validation_only"] = True
        config_path = CONFIG_DIR / f"{run_id}.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=4, ensure_ascii=False)
            handle.write("\n")

        relative_config = config_path.relative_to(ROOT / "configurations")
        command = [
            str(PYTHON),
            "-u",
            "run.py",
            "--config_file",
            str(relative_config).replace("\\", "/"),
            "--seed",
            "1",
            "--run_id",
            run_id,
        ]
        print(
            f"\n=== START {run_id}: kd_alpha={alpha}, lr={learning_rate} ===",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"=== COMPLETE {run_id} ===", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
