"""Run the ICDE teacher-independence check on one non-retrieval teacher.

The experiment deliberately separates three artifacts:
1. ``student_init_ckpt`` supplies architecture-compatible weights for the
   activation-aware low-rank initialization;
2. ``teacher_predictions_path`` supplies only soft targets through the public
   prediction-cache contract;
3. labels from the official training split supply the supervised objective.

The selected teacher is the existing PEMS04 AGCRN-spatial checkpoint. It has
no Hist2Res retrieval, exact-history, or reranker branch. Validation selects
the student checkpoint; Test is evaluated once by the standard runner.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(sys.executable)
TEACHER_EXP = (
    ROOT / "EXP_experiments" / "TLAST_improved" / "PEMS04"
    / "TLAST_improved_PEMS04_12to12_exp07131820_s1"
)
CONFIG = (
    ROOT / "configurations" / "icde_generality"
    / "pems04_agcrn_teacher_lite_r256_s1.json"
)
ARTIFACT_DIR = ROOT / "paper_results" / "icde_generality"
TEACHER_CACHE = ARTIFACT_DIR / "pems04_agcrn_teacher_predictions_s1.npz"
ACTIVATION_CACHE = (
    ARTIFACT_DIR / "pems04_agcrn_teacher_activation_second_moment_s1.npz"
)
RUN_ID = "icde_pems04_agcrn_teacher_lite_r256_s1"
EXP_DIR = (
    ROOT / "EXP_experiments" / "TLAST_improved" / "PEMS04"
    / f"TLAST_improved_PEMS04_12to12_exp{RUN_ID}"
)


def invoke(command: list[str], log_path: Path, dry_run: bool) -> None:
    print("[RUN]", subprocess.list2cmdline(command), flush=True)
    if dry_run:
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


def training_complete() -> bool:
    if not (EXP_DIR / "best.pth").is_file():
        return False
    return any(
        "Training finished" in path.read_text(
            encoding="utf-8", errors="ignore"
        )
        for path in EXP_DIR.glob("*.log")
    )


def validate_config() -> None:
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model = config["model_config"]
    training = config["training_config"]
    required = {
        "lowrank_student_finetune": 1,
        "student_include_retrieval_adapters": 0,
        "teacher_predictions_strict_alignment": 1,
    }
    for key, expected in required.items():
        if model.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if not training.get("validation_only", False):
        raise ValueError("The second-teacher run must be validation_only")
    if any(
        model.get(key, 0)
        for key in (
            "use_analog_retrieval_residual",
            "use_analog_lowrank_retrieval_residual",
            "use_analog_exact_match_residual",
        )
    ):
        raise ValueError("The second teacher control must be retrieval-free")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device used to dump activations and teacher predictions.",
    )
    parser.add_argument(
        "--python", type=Path, default=DEFAULT_PYTHON,
        help="Python executable for the CUDA experiment environment.",
    )
    args = parser.parse_args()

    validate_config()
    if not (TEACHER_EXP / "best.pth").is_file():
        raise FileNotFoundError(TEACHER_EXP / "best.pth")
    if not args.dry_run and not args.python.is_file():
        raise FileNotFoundError(args.python)
    python = str(args.python)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEACHER_CACHE.is_file():
        invoke(
            [
                python,
                "dump_teacher_predictions.py",
                "--exp_dir", str(TEACHER_EXP),
                "--output", str(TEACHER_CACHE),
                "--teacher-name", "AGCRN-spatial non-retrieval teacher",
            ],
            ROOT / "_icde_second_teacher_dump.log",
            args.dry_run,
        )
    else:
        print(f"[SKIP] teacher cache exists: {TEACHER_CACHE}")

    if not ACTIVATION_CACHE.is_file():
        invoke(
            [
                python,
                "compute_predictor_activation_covariance.py",
                "--exp-dir", str(TEACHER_EXP),
                "--output", str(ACTIVATION_CACHE),
                "--device", args.device,
            ],
            ROOT / "_icde_second_teacher_activation.log",
            args.dry_run,
        )
    else:
        print(f"[SKIP] activation cache exists: {ACTIVATION_CACHE}")

    if not training_complete():
        relative_config = CONFIG.relative_to(ROOT / "configurations")
        invoke(
            [
                python,
                "-u",
                "run.py",
                "--config_file", str(relative_config).replace("\\", "/"),
                "--seed", "1",
                "--run_id", RUN_ID,
                "--device", args.device,
            ],
            ROOT / "_icde_second_teacher_train.log",
            args.dry_run,
        )
    else:
        print(f"[SKIP] completed experiment exists: {EXP_DIR}")


if __name__ == "__main__":
    main()
