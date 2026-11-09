"""Five fresh-process batch-1 latency runs for frozen E2 budget points.

Each reported measurement is one warmed full-Validation online replay in a
separate Python process.  Order is randomized within each round, and raw
benchmark JSONs are retained to expose process-to-process variance.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "latency_repeats_v1"
BENCH = ROOT / "paper_results/icde_retrieval_free_pilot/benchmark_online_end_to_end.py"
CASES = {
    "PEMS04": {
        "exp": "EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_exppems04_joint3_lite_r256_strictalign_s1",
        "full": "cold_start_streaming/pems04_full_cold_state.npz",
    },
    "PEMS08": {
        "exp": "EXP_experiments/TLAST_improved/PEMS08/TLAST_improved_PEMS08_12to12_exppems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1",
        "full": "cold_start_streaming/pems08_full_cold_state.npz",
    },
}
VARIANTS = ("backbone", "full", "top_gain_10pct", "ordinary_gain_per_byte_10pct",
            "top_gain_40pct", "ordinary_gain_per_byte_40pct")
ROUNDS = 5


def state_for(dataset, cfg, variant):
    if variant == "backbone":
        return None
    if variant == "full":
        return ROOT / cfg["full"]
    return ROOT / "icde_e123/e2" / dataset.lower() / f"{variant}_cold_state.npz"


def run(dataset, cfg, variant, round_index):
    folder = OUT / dataset.lower()
    folder.mkdir(parents=True, exist_ok=True)
    tag = f"round{round_index:02d}_{variant}"
    result = folder / f"{tag}.json"
    state = state_for(dataset, cfg, variant)
    if state is not None and not state.is_file():
        raise FileNotFoundError(state)
    if not result.exists():
        cmd = [sys.executable, "-B", str(BENCH),
               "--exp-dir", str(ROOT / cfg["exp"]),
               "--mode", "sparse" if variant not in ("backbone", "full") else variant,
               "--batch-size", "1", "--warmup-passes", "1", "--repeats", "1",
               "--output", str(result)]
        if state is not None:
            cmd.extend(("--state", str(state)))
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "_runtime_deps"), str(ROOT),
                                               env.get("PYTHONPATH", "")))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (folder / f"{tag}.stdout.log").open("w", encoding="utf-8") as stdout, \
             (folder / f"{tag}.stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout,
                                       stderr=stderr, creationflags=flags, check=False)
        if completed.returncode or not result.is_file():
            raise RuntimeError(f"{dataset} {tag} failed; inspect stderr log")
    row = json.loads(result.read_text(encoding="utf-8"))
    if (row["test_evaluated"] or row["query_cache_loaded"] or row["batch_size"] != 1
            or row["warmup_passes"] != 1 or row["repeats"] != 1
            or not row["device"].startswith("cuda")):
        raise ValueError(f"{dataset} {tag}: protocol mismatch")
    if state is not None and row["state_bytes"] != state.stat().st_size:
        raise ValueError(f"{dataset} {tag}: state byte mismatch")
    print(f"{dataset} round {round_index}/{ROUNDS}: {variant} done", flush=True)
    return row


def median_and_range(rows, key):
    values = [float(row[key]) for row in rows]
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "individual": values}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {dataset: {variant: [] for variant in VARIANTS} for dataset in CASES}
    for round_index in range(1, ROUNDS + 1):
        tasks = [(dataset, variant) for dataset in CASES for variant in VARIANTS]
        random.Random(20261015 + round_index).shuffle(tasks)
        for dataset, variant in tasks:
            rows[dataset][variant].append(run(dataset, CASES[dataset], variant, round_index))
    summary = {
        "protocol": "E2_independent_fresh_process_latency_v1",
        "split": "full Validation, online replay, no Test",
        "rounds": ROUNDS,
        "order": "fixed-seed randomized within round",
        "per_process": "batch=1, one warmup pass, one measured pass",
        "datasets": {},
    }
    for dataset, variants in rows.items():
        summary["datasets"][dataset] = {}
        for variant, samples in variants.items():
            summary["datasets"][dataset][variant] = {
                "state_bytes": samples[0]["state_bytes"],
                "selected_nodes": samples[0]["selected_nodes"],
                "samples_per_pass": samples[0]["samples_per_pass"],
                "pass_wall_seconds": median_and_range(samples, "pass_wall_median_seconds"),
                "end_to_end_batch_latency_ms_p50": median_and_range(
                    samples, "end_to_end_batch_latency_ms_p50"),
                "end_to_end_batch_latency_ms_p95": median_and_range(
                    samples, "end_to_end_batch_latency_ms_p95"),
                "retrieval_seconds": median_and_range(samples, "retrieval_median_seconds"),
            }
    destination = OUT / "summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
