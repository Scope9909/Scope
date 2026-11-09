"""Audit frozen SRM memory and consolidate existing accuracy/timing evidence.

The eight memory loads run in isolated processes. No prediction or Test target
is read, and all accuracy and timing numbers come from completed prior runs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np

DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")
MODES = ("sparse", "full")
ARRAY_GROUPS = {
    "observations": ("train_raw",),
    "candidate_indices": ("idx_train", "train_tod", "pending_idx_train",
                          "pending_train_tod", "_train_positions_by_tod"),
    "scales_and_nodes": ("node_ids", "quant_scale", "scaler_mean", "scaler_std"),
    "fallback_statistics": ("_global_sum", "_global_count", "_tod_sum",
                            "_tod_count", "_global_mean", "_tod_mean"),
    "offsets": ("x_offsets", "y_offsets"),
}


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
                ("private_usage", ctypes.c_size_t)]


ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (
    ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD)
ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def rss_bytes() -> int:
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError()
    return int(counters.working_set_size)


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def arrays_in(value):
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from arrays_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from arrays_in(item)


def owner(array: np.ndarray) -> np.ndarray:
    while isinstance(array.base, np.ndarray):
        array = array.base
    return array


def memory_worker(dataset: str, mode: str, state: Path) -> None:
    from utils.online_residual_memory import OnlineResidualMemory

    if not state.is_file():
        raise FileNotFoundError(state)
    before = rss_bytes()
    samples = [before]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.005):
            samples.append(rss_bytes())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        memory = OnlineResidualMemory.from_npz(state)
        after = rss_bytes()
    finally:
        stop.set()
        sampler.join()
    seen = set()
    group_bytes = {}
    group_arrays = {}
    for group, names in ARRAY_GROUPS.items():
        size = count = 0
        for name in names:
            for array in arrays_in(getattr(memory, name)):
                base = owner(array)
                key = id(base)
                if key not in seen:
                    seen.add(key)
                    size += int(base.nbytes)
                    count += 1
        group_bytes[group] = size
        group_arrays[group] = count
    index = memory._train_positions_by_tod
    result = {
        "protocol": "frozen_srm_cold_start_memory_ledger_v1",
        "dataset": dataset, "mode": mode, "state": str(state),
        "state_file_bytes": state.stat().st_size,
        "nodes": int(len(memory.node_ids)),
        "active_candidates": int(len(memory.idx_train)),
        "pending_candidates": memory.pending_count,
        "rss_before_state_load_bytes": before,
        "rss_after_state_load_bytes": after,
        "rss_increase_after_load_bytes": after - before,
        "rss_peak_during_load_sampled_bytes": max(samples + [after]),
        "rss_peak_increase_sampled_bytes": max(samples + [after]) - before,
        "rss_samples": len(samples),
        "retained_numpy_payload_by_group_bytes": group_bytes,
        "retained_numpy_arrays_by_group": group_arrays,
        "retained_numpy_payload_total_bytes": sum(group_bytes.values()),
        "tod_index_buckets": len(index),
        "tod_index_dict_shallow_bytes": sys.getsizeof(index),
        "note": "RSS is a process working set, sampled during cold-start load. "
                "It includes Python/runtime allocations and is not SRM-exclusive RAM. "
                "NumPy payload deduplicates shared base arrays and excludes transient buffers.",
    }
    write_json(OUT / f"{dataset.lower()}_{mode}_memory.json", result)


def source_data() -> tuple[dict, dict, dict]:
    diagnostics = json.loads((ROOT / "srm_validation_diagnostics" /
                              "results.json").read_text(encoding="utf-8"))
    repeats = json.loads((ROOT / "srm_followup_validation" /
                          "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "srm_test_once" /
                           "frozen_manifest.json").read_text(encoding="utf-8"))
    if diagnostics["status"] != "complete" or repeats["status"] != "complete":
        raise ValueError("Prerequisite Validation runs are incomplete")
    if len(repeats["jobs"]) != 40 or len(manifest["jobs"]) != 36:
        raise ValueError("Prerequisite run count differs from frozen protocol")
    return diagnostics, repeats, manifest


def report(rows: list[dict]) -> str:
    lines = ["# SRM 完整内存账本与现有实验指标", "",
             "四个数据集、sparse/full 各一次独立冷启动加载。精度和五次计时来自已有结果；本轮没有重复预测或读取 Test 标签。", "",
             "| 数据集 | 模式 | Test 状态 MiB | checkpoint+校准+状态 MiB | Validation 常驻 NumPy MiB | 冷启动 RSS 增量 MiB | 检索中位秒 | 全程中位秒 | Test MAE |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    mib = 1024 ** 2
    for row in rows:
        m = row["memory"]
        lines.append(
            f"| {row['dataset']} | {row['mode']} | {row['test_state_bytes']/mib:.3f} | "
            f"{row['deployment_disk_bytes']/mib:.3f} | "
            f"{m['retained_numpy_payload_total_bytes']/mib:.3f} | "
            f"{m['rss_increase_after_load_bytes']/mib:.1f} | "
            f"{row['retrieval_median_seconds']:.2f} | "
            f"{row['wall_median_seconds']:.2f} | "
            f"{row['test_fixed_calibrated_mae_mean']:.6f} |")
    lines += ["", "口径：Test 状态与 checkpoint+校准+状态是磁盘文件字节；常驻 NumPy 分项和 RSS 增量来自 Validation 冷启动状态。完整分组、五次计时范围及原始来源见 `ledger.json`。", "",
              "RSS 是进程工作集差值，包含运行时和加载器分配，不能直接称为检索模块独占内存。冷启动峰值按 5 ms 采样，可能漏掉更短的瞬时峰值。", "",
              "PEMS03/08 的三个 seed 标签共用 checkpoint，表中的 Test MAE 均值不代表三次独立训练。batch=32 顺序重放时间也不等于 batch=1 单请求时延。", ""]
    return "\n".join(lines)


def run_queue() -> None:
    diagnostics, repeats, manifest = source_data()
    rows = []
    manifest_jobs = {(job["dataset"], job["seed"], job["mode"]): job
                     for job in manifest["jobs"]}
    for dataset in DATASETS:
        for mode in MODES:
            print(f"{datetime.now().isoformat(timespec='seconds')} {dataset} {mode}", flush=True)
            state = Path(diagnostics["preflight"][dataset][f"{mode}_state"])
            output = OUT / f"{dataset.lower()}_{mode}_memory.json"
            if output.exists():
                raise FileExistsError(output)
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                            "--worker", "--dataset", dataset, "--mode", mode,
                            "--state", str(state)], cwd=ROOT, check=True)
            memory = json.loads(output.read_text(encoding="utf-8"))
            jobs = [job for job in repeats["jobs"] if job["dataset"] == dataset
                    and job["mode"] == mode]
            if sorted(job["repeat"] for job in jobs) != [1, 2, 3, 4, 5]:
                raise ValueError(f"Incomplete timing repeats: {dataset}/{mode}")
            test_rows = []
            for seed in (1, 2, 3):
                path = (ROOT / "srm_test_once" /
                        f"{dataset.lower()}_s{seed}_{mode}_test_once.json")
                test = json.loads(path.read_text(encoding="utf-8"))
                if test["test_evaluations"] != 1 or test["test_based_selection"]:
                    raise ValueError(f"Unexpected Test protocol: {path}")
                test_rows.append(test)
            frozen = manifest_jobs[(dataset, 1, mode)]
            disk_bytes = sum(Path(frozen[key]).stat().st_size
                             for key in ("checkpoint", "calibration", "test_state"))
            row = {
                "dataset": dataset, "mode": mode,
                "memory": memory,
                "test_state_bytes": test_rows[0]["test_state_bytes"],
                "deployment_disk_bytes": disk_bytes,
                "checkpoint_disk_bytes": Path(frozen["checkpoint"]).stat().st_size,
                "calibration_disk_bytes": Path(frozen["calibration"]).stat().st_size,
                "validation_raw_mae": test_rows[0]["validation_raw"]["mae"],
                "test_fixed_calibrated_mae_mean": statistics.mean(
                    test["test_fixed_calibrated"]["mae"] for test in test_rows),
                "test_checkpoint_hashes": [test["checkpoint_sha256"] for test in test_rows],
                "retrieval_seconds": [job["retrieval_seconds"] for job in jobs],
                "retrieval_median_seconds": statistics.median(
                    job["retrieval_seconds"] for job in jobs),
                "wall_seconds": [job["wall_seconds"] for job in jobs],
                "wall_median_seconds": statistics.median(job["wall_seconds"] for job in jobs),
                "maintenance_seconds_existing_run": diagnostics["results"][dataset]
                    ["modes"][mode]["maintenance_seconds_total"],
                "validation_array_payload_start_existing_run": diagnostics["results"][dataset]
                    ["modes"][mode]["array_payload_start_bytes"],
            }
            rows.append(row)
    write_json(OUT / "ledger.json", {
        "status": "complete", "completed_at": datetime.now().astimezone().isoformat(),
        "protocol": "frozen_srm_memory_ledger_reusing_existing_metrics_v1",
        "rows": rows,
        "sources": ["srm_validation_diagnostics/results.json",
                    "srm_followup_validation/status.json",
                    "srm_test_once/frozen_manifest.json"],
    })
    path = OUT / "SRM_MEMORY_LEDGER_20261015.md"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(report(rows), encoding="utf-8")
    print("complete 8/8", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    if args.preflight:
        diagnostics, repeats, manifest = source_data()
        for dataset in DATASETS:
            for mode in MODES:
                state = Path(diagnostics["preflight"][dataset][f"{mode}_state"])
                if not state.is_file():
                    raise FileNotFoundError(state)
                jobs = [job for job in repeats["jobs"] if job["dataset"] == dataset
                        and job["mode"] == mode]
                if sorted(job["repeat"] for job in jobs) != [1, 2, 3, 4, 5]:
                    raise ValueError(f"Incomplete repeats: {dataset}/{mode}")
                for seed in (1, 2, 3):
                    path = (ROOT / "srm_test_once" /
                            f"{dataset.lower()}_s{seed}_{mode}_test_once.json")
                    if not path.is_file():
                        raise FileNotFoundError(path)
        print(f"preflight passed: {len(manifest['jobs'])} frozen Test records, "
              f"{len(repeats['jobs'])} timing records, 8 states")
    elif args.worker:
        if not args.dataset or not args.mode or not args.state:
            parser.error("Worker requires dataset, mode and state")
        memory_worker(args.dataset, args.mode, args.state)
    else:
        if (OUT / "ledger.json").exists():
            raise FileExistsError("Ledger already completed")
        run_queue()


if __name__ == "__main__":
    main()
