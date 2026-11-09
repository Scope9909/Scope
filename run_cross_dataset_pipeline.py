"""Seed-parameterised cross-dataset pipeline driver.

Python port of ``run_cross_dataset_experiments.ps1``.  It exists because long
training runs have to be hosted by a plain Python process (a PowerShell host is
not stable enough here), while keeping exactly the same semantics:

    backbone -> [caches] -> lowrank -> exact -> reranker_lowrank
             -> head -> confidence -> final_joint -> collect results

Stage completion is recorded with a ``.pipeline_complete`` marker next to
``best.pth``.  A run directory that has ``best.pth`` but no marker is treated as
a *partial* run and aborts the pipeline instead of silently reusing it.

Retrieval / candidate / index-scaler caches are built from data only (no learned
parameters), so they are shared across seeds: for any dataset whose caches
already exist the cache step is a no-op regardless of ``--seed``.

Usage
-----
    python run_cross_dataset_pipeline.py --datasets PEMS08 --seed 2
    python run_cross_dataset_pipeline.py --datasets PEMS08 --seed 2 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGES = (
    "backbone",
    "lowrank",
    "exact",
    "reranker_lowrank",
    "head",
    "confidence",
    "final_joint",
)
SUPPORTED_DATASETS = ("PEMS03", "PEMS07", "PEMS08", "PEMSBAY")


def run_id(dataset: str, seed: int, stage: str) -> str:
    return f"paper_{dataset.lower()}_s{seed}_{stage}"


def experiment_dir(dataset: str, seed: int, stage: str) -> Path:
    return (
        ROOT
        / "EXP_experiments"
        / "TLAST_improved"
        / dataset
        / f"TLAST_improved_{dataset}_12to12_exp{run_id(dataset, seed, stage)}"
    )


def call(command: list[str], *, dry_run: bool) -> int:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] [CMD] {' '.join(command)}", flush=True)
    if dry_run:
        return 0
    return subprocess.call(command, cwd=str(ROOT))


def ensure_retrieval_caches(
    dataset: str, python_exe: str, *, dry_run: bool
) -> None:
    cache_dir = ROOT / "datasets" / dataset
    time_cache = cache_dir / f"{dataset}_time.npz"
    index_cache = cache_dir / f"{dataset}_12to12_index_scaler.npz"
    if not time_cache.exists():
        raise FileNotFoundError(f"Backbone did not create time cache: {time_cache}")
    if not index_cache.exists():
        raise FileNotFoundError(f"Backbone did not create index cache: {index_cache}")

    base = cache_dir / f"{dataset}_12to12_analog_k8.npz"
    exact = cache_dir / f"{dataset}_12to12_analog_k8_exact_mean.npz"
    candidate = cache_dir / f"{dataset}_12to12_analog_k8_candidate_indices.npz"

    if base.exists():
        print(f"[CACHE] skip base retrieval (exists): {dataset}", flush=True)
    else:
        code = call(
            [
                python_exe,
                "compute_analog_retrieval_cache.py",
                "--dataset_name", dataset,
                "--input_length", "12",
                "--predict_length", "12",
                "--k", "8",
                "--min_valid_history", "6",
                "--aggregation", "mean",
                "--output", str(base.relative_to(ROOT)).replace("\\", "/"),
            ],
            dry_run=dry_run,
        )
        if code != 0:
            raise RuntimeError(f"Base retrieval cache failed ({code}): {dataset}")

    if exact.exists():
        print(f"[CACHE] skip exact retrieval (exists): {dataset}", flush=True)
    else:
        code = call(
            [
                python_exe,
                "compute_analog_exact_match_cache.py",
                "--dataset_name", dataset,
                "--input_length", "12",
                "--predict_length", "12",
                "--min_valid_history", "6",
                "--base_cache", str(base.relative_to(ROOT)).replace("\\", "/"),
                "--output", str(exact.relative_to(ROOT)).replace("\\", "/"),
            ],
            dry_run=dry_run,
        )
        if code != 0:
            raise RuntimeError(f"Exact retrieval cache failed ({code}): {dataset}")

    if candidate.exists():
        print(f"[CACHE] skip candidate indices (exists): {dataset}", flush=True)
    else:
        code = call(
            [
                python_exe,
                "compute_analog_candidate_index_cache.py",
                "--dataset_name", dataset,
                "--input_length", "12",
                "--predict_length", "12",
                "--k", "8",
                "--min_valid_history", "6",
                "--reference_cache", str(exact.relative_to(ROOT)).replace("\\", "/"),
                "--output", str(candidate.relative_to(ROOT)).replace("\\", "/"),
            ],
            dry_run=dry_run,
        )
        if code != 0:
            raise RuntimeError(f"Candidate cache failed ({code}): {dataset}")


def invoke_stage(
    dataset: str, seed: int, stage: str, python_exe: str, *, dry_run: bool
) -> None:
    rid = run_id(dataset, seed, stage)
    target = experiment_dir(dataset, seed, stage)
    best = target / "best.pth"
    complete = target / ".pipeline_complete"

    if best.exists() and complete.exists():
        print(f"[SKIP] completed: {rid}", flush=True)
        return
    if target.exists():
        raise RuntimeError(
            f"Incomplete run directory exists: {target}. Archive or remove it "
            "before resuming so a partial best.pth is not reused."
        )

    config = f"cross_dataset/{dataset.lower()}_s{seed}_{stage}.json"
    code = call(
        [
            python_exe, "run.py",
            "--config_file", config,
            "--seed", str(seed),
            "--run_id", rid,
        ],
        dry_run=dry_run,
    )
    if code != 0:
        raise RuntimeError(f"Experiment failed with exit code {code}: {rid}")
    if dry_run:
        return
    if not best.exists():
        raise RuntimeError(f"Experiment ended without best.pth: {rid}")
    complete.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["PEMS08"], choices=SUPPORTED_DATASETS
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--stages", nargs="+", default=list(STAGES), choices=list(STAGES)
    )
    parser.add_argument("--skip-config-gen", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.skip_config_gen:
        code = call(
            [
                args.python_exe,
                "make_cross_dataset_experiment_configs.py",
                "--datasets", *args.datasets,
                "--seed", str(args.seed),
            ],
            dry_run=args.dry_run,
        )
        if code != 0:
            raise RuntimeError(f"Config generation failed with exit code {code}")

    for dataset in args.datasets:
        if "backbone" not in args.stages:
            ensure_retrieval_caches(dataset, args.python_exe, dry_run=args.dry_run)
        for stage in args.stages:
            invoke_stage(
                dataset, args.seed, stage, args.python_exe, dry_run=args.dry_run
            )
            if stage == "backbone":
                ensure_retrieval_caches(
                    dataset, args.python_exe, dry_run=args.dry_run
                )
        if not args.skip_collect:
            code = call(
                [args.python_exe, "collect_paper_results.py", "--dataset", dataset],
                dry_run=args.dry_run,
            )
            if code != 0:
                raise RuntimeError(f"Result collection failed ({code}): {dataset}")

    print("[DONE] pipeline finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
