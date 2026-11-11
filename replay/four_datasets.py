"""Replay frozen PEMS03/04/07/08 configurations, Validation before Test."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replay import state_codec  # noqa: E402
from replay.calibration import fit_and_apply, metrics  # noqa: E402
from replay.runtime import (  # noqa: E402
    build_runtime,
    collect,
)

DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=DATASETS, action="append",
        help="Replay one or more datasets; default: all four",
    )
    return parser.parse_args()


def replay(dataset: str, split: str, output_dir: Path) -> dict:
    from utils.online_residual_memory import OnlineResidualMemory

    loaders, scaler, model, device, index_path, calibration, selector = (
        build_runtime(dataset)
    )
    with np.load(index_path, allow_pickle=False) as archive:
        endpoints = np.asarray(archive[f"idx_{split}"], dtype=np.int64)
    state_file = ROOT / "assets/states" / dataset / (
        "validation.alsc" if split == "val" else "test.alsc"
    )
    memory = OnlineResidualMemory(state_codec.load_runtime_state(state_file))
    if split == "val" and memory.cold_start_endpoint != int(endpoints[0]):
        raise ValueError(f"Validation state alignment mismatch for {dataset}")
    if split == "test" and memory.cold_start_endpoint is not None:
        raise ValueError(f"Test state is not fully observed for {dataset}")

    started = time.perf_counter()
    prediction, truth = collect(
        loaders[f"{split}_loader"], endpoints, model, scaler, memory, device, 32
    )
    if len(truth) != len(endpoints):
        raise AssertionError(f"Repeated padding samples included for {dataset} {split}")
    row = {
        "dataset": dataset,
        "split": split,
        "seed": 1,
        "selector": selector,
        "device": str(device),
        "sample_count": len(truth),
        "raw_metrics": metrics(prediction, truth),
        "state_file": state_file.relative_to(ROOT).as_posix(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    validation_file = output_dir / f"{dataset.lower()}_validation.npz"
    if split == "val":
        np.savez_compressed(
            validation_file, prediction=prediction, truth=truth, endpoints=endpoints
        )
    else:
        if not validation_file.is_file():
            raise FileNotFoundError(f"Run Validation first; missing {validation_file}")
        with np.load(validation_file, allow_pickle=False) as archive:
            val_prediction = np.asarray(archive["prediction"], dtype=np.float64)
            val_truth = np.asarray(archive["truth"], dtype=np.float64)
            val_endpoints = np.asarray(archive["endpoints"], dtype=np.int64)
        parameters = json.loads(calibration.read_text(encoding="utf-8"))
        kwargs = {
            key: parameters[key]
            for key in (
                "num_bins", "smooth_shrinkage", "num_time_bins",
                "time_shrinkage", "slices_per_day",
            )
        }
        _, calibrated, _ = fit_and_apply(
            val_prediction,
            val_truth,
            val_endpoints,
            prediction,
            endpoints,
            **kwargs,
        )
        row["calibrated_metrics"] = metrics(calibrated, truth)
    (output_dir / f"{dataset.lower()}_{split}_metrics.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8"
    )
    return row


def main() -> None:
    cli = parse_args()
    output_dir = cli.output_dir.resolve()
    if cli.split == "val":
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not output_dir.is_dir():
        raise FileNotFoundError("Run Validation first with the same output directory")
    datasets = tuple(cli.dataset or DATASETS)
    rows = []
    for dataset in datasets:
        print(f"{dataset} {cli.split} replay starting", flush=True)
        row = replay(dataset, cli.split, output_dir)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {"split": cli.split, "complete": len(rows) == len(datasets), "results": rows}
    (output_dir / f"{cli.split}_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
