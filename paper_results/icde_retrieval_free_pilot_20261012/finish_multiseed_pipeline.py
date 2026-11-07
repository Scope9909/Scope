"""Wait for PEMS04/07/08, then automatically prepare/run PEMS03 and summarize."""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / "_runtime_deps" / "python311_embed" / "python.exe"


def process_running(pid: int) -> bool | None:
    if pid <= 0:
        return None
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x00100000, 0, pid)
    if not handle:
        return None
    try:
        return kernel.WaitForSingleObject(handle, 0) == 258
    finally:
        kernel.CloseHandle(handle)


def status(path: Path, stage: str, **extra) -> None:
    path.write_text(json.dumps({
        "protocol": "four_dataset_three_seed_validation_pipeline_v1",
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": False,
        **extra,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_logged(name: str, arguments: list[str], out: Path) -> None:
    log = out / f"{name}.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([str(PYTHON), "-B", *arguments], cwd=ROOT,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed; inspect {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed1-dir", type=Path, required=True)
    parser.add_argument("--queue-pid", type=int, default=0)
    parser.add_argument("--max-wait-hours", type=float, default=6)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "pipeline_status.json"
    expected = [out / f"{dataset}_s{seed}_{mode}_b1_r3.json"
                for dataset in ("pems04", "pems07", "pems08")
                for seed in (2, 3)
                for mode in ("backbone", "sparse", "full")]
    try:
        status(status_path, "waiting_for_pems04_pems07_pems08", completed=sum(
            path.is_file() for path in expected), planned=len(expected))
        deadline = time.monotonic() + args.max_wait_hours * 3600
        while not all(path.is_file() for path in expected):
            if time.monotonic() >= deadline:
                raise TimeoutError("First three datasets did not finish within the wait window")
            if process_running(args.queue_pid) is False:
                missing = [path.name for path in expected if not path.is_file()]
                raise RuntimeError(f"First queue exited with missing results: {missing}")
            time.sleep(60)
        status(status_path, "running_pems03", completed=len(expected), planned=len(expected))
        run_logged("pems03_automatic_continuation", [
            str(HERE / "prepare_and_run_pems03_multiseed.py"),
            "--output-dir", str(out),
        ], out)
        status(status_path, "summarizing")
        summary = out / "four_dataset_three_seed_summary.json"
        report = out / "FOUR_DATASET_THREE_SEED_VALIDATION.md"
        run_logged("four_dataset_summary", [
            str(HERE / "summarize_multiseed_single_request.py"),
            "--seed1-dir", str(args.seed1_dir.resolve()),
            "--multiseed-dir", str(out),
            "--output-json", str(summary), "--output-md", str(report),
        ], out)
        status(status_path, "complete", report=str(report), summary=str(summary),
               results=36)
        print(f"Complete: {report}", flush=True)
    except BaseException as error:
        status(status_path, "failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
