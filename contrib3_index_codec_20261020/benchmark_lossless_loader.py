"""Independent-process load/decode benchmark for frozen Test state packages.

This reads state files only. It does not load predictions, labels, or run Test
inference. Results describe warm filesystem-cache behavior unless the operator
independently controls the OS cache.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import adaptive_lossless_state_codec as codec
from online_codec_runtime import decode_online_state
from pilot_reversible_predictive_coding import normalize_state, best_bytes

ONLINE = HERE / "packages/online"
TEST_DIR = ONLINE / "adaptive_lossless_test_20261023_v1"
KINDS = ("legacy_npz", "generic", "exact", "selected")


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def memory_counters():
    counter = MemoryCounters()
    counter.cb = ctypes.sizeof(counter)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counter), counter.cb)
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return counter.WorkingSetSize, counter.PeakWorkingSetSize


def worker(path: Path, kind: str):
    before, _ = memory_counters()
    if kind == "legacy_npz":
        t0 = time.perf_counter()
        blob = path.read_bytes()
        read_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        state = decode_online_state(io.BytesIO(blob))
        state["train_raw"] = state["train_raw"].astype(np.float32)
        decode_s = time.perf_counter() - t1
    else:
        # Measure the actual deployment-facing entry point directly. It reads
        # the archive once internally; do not pre-read it in the benchmark.
        read_s = 0.0
        t1 = time.perf_counter()
        state = codec.load_runtime_state(path)
        decode_s = time.perf_counter() - t1
    after, peak = memory_counters()
    result = dict(path=str(path.resolve()), kind=kind, file_bytes=path.stat().st_size,
                  read_seconds=read_s, decode_seconds=decode_s,
                  total_seconds=read_s + decode_s,
                  working_set_before=before, working_set_after=after,
                  peak_working_set=peak,
                  array_nbytes=sum(np.asarray(value).nbytes for value in state.values()),
                  train_raw_shape=list(state["train_raw"].shape),
                  train_raw_dtype=str(state["train_raw"].dtype))
    print(json.dumps(result), flush=True)


def prepare_legacy(output: Path):
    paths = {}
    for dataset in ("pems03", "pems04", "pems07", "pems08"):
        exact, _ = normalize_state(ONLINE / f"{dataset}_s1_test_u16.npz")
        blob, _ = best_bytes(exact)
        path = output / f"{dataset}_test_legacy_npz.npz"
        path.write_bytes(blob)
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            if not np.array_equal(archive["train_raw"], exact["train_raw"]):
                raise AssertionError("Legacy baseline differs from exact history")
        paths[dataset] = path
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--kind", choices=KINDS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.kind)
        return
    if args.output is None or args.repeats < 1:
        parser.error("--output and positive --repeats are required")
    args.output.mkdir(parents=True, exist_ok=False)
    legacy = prepare_legacy(args.output)
    raw = []
    # Rotate methods on each repeat so process startup/caching trends are less
    # tightly coupled to one method. OS disk cache remains uncontrolled.
    for repeat in range(args.repeats):
        kinds = KINDS[repeat % len(KINDS):] + KINDS[:repeat % len(KINDS)]
        for dataset in ("pems03", "pems04", "pems07", "pems08"):
            for kind in kinds:
                path = (legacy[dataset] if kind == "legacy_npz" else
                        TEST_DIR / f"{dataset}_test_{kind}.alsc")
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()),
                     "--worker", str(path), "--kind", kind],
                    capture_output=True, text=True, check=True)
                row = json.loads(proc.stdout.strip())
                row.update(dataset=dataset.upper(), repeat=repeat + 1,
                           file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                raw.append(row)
                print(dataset, kind, repeat + 1,
                      round(row["total_seconds"] * 1000, 2), "ms", flush=True)
                (args.output / "raw.json").write_text(
                    json.dumps(raw, indent=2), encoding="utf-8")
    summary = []
    for dataset in ("pems03", "pems04", "pems07", "pems08"):
        for kind in KINDS:
            rows = [r for r in raw if r["dataset"] == dataset.upper() and r["kind"] == kind]
            summary.append(dict(dataset=dataset.upper(), kind=kind,
                                file_bytes=rows[0]["file_bytes"], trials=len(rows),
                                median_read_ms=1000 * statistics.median(r["read_seconds"] for r in rows),
                                median_decode_ms=1000 * statistics.median(r["decode_seconds"] for r in rows),
                                median_total_ms=1000 * statistics.median(r["total_seconds"] for r in rows),
                                min_total_ms=1000 * min(r["total_seconds"] for r in rows),
                                max_total_ms=1000 * max(r["total_seconds"] for r in rows),
                                median_after_working_set_mib=statistics.median(
                                    r["working_set_after"] for r in rows) / 2**20,
                                median_peak_working_set_mib=statistics.median(
                                    r["peak_working_set"] for r in rows) / 2**20,
                                decoded_array_nbytes=rows[0]["array_nbytes"]))
    report = dict(scope="State load/decode only; no prediction or labels",
                  cold_disk_claim=False, repeats=args.repeats, summary=summary)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
