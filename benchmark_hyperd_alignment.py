"""Unified efficiency benchmark aligned with the efficiency axes in HyperD.

This script never trains or changes a checkpoint.  It loads one completed
experiment, runs a fixed split repeatedly, and reports:

* neural parameter count;
* forward-only and batch end-to-end latency;
* throughput and CPU/GPU peak memory;
* checkpoint, retrieval-cache, candidate-cache and calibrator storage;
* optional torch-profiler FLOPs for supported operators (a lower bound);
* optional frozen-calibrator CPU latency from an existing prediction archive.

Run every model in a fresh process.  That makes CPU RSS and CUDA peak-memory
numbers comparable and prevents one model's allocator/cache state from leaking
into another model's measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def _logger() -> logging.Logger:
    logger = logging.getLogger("hyperd_alignment_benchmark")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    return logger


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _file_record(path: Path | None, role: str) -> dict[str, Any] | None:
    if path is None:
        return None
    record: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        record["bytes"] = int(path.stat().st_size)
        record["mib"] = float(path.stat().st_size / (1024**2))
    return record


def _prepare_batch(
    batch: Any,
    device: torch.device,
    unpack_dataloader_input: Any,
) -> tuple[list[torch.Tensor], torch.Tensor, int]:
    loader_input, loader_target = batch
    (
        x,
        x_time,
        analog_retrieval,
        exact_retrieval,
        exact_available,
        candidate_future,
        candidate_available,
        candidate_features,
    ) = unpack_dataloader_input(loader_input)
    y, y_time = loader_target
    model_input = [
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(x_time, dtype=torch.long, device=device),
        torch.as_tensor(y_time, dtype=torch.long, device=device),
    ]
    for value in (
        analog_retrieval,
        exact_retrieval,
        exact_available,
        candidate_future,
        candidate_available,
        candidate_features,
    ):
        if value is None:
            continue
        # CPU kernels used by this project are not uniformly implemented for
        # float16.  Cast cache payloads to float32 for the CPU benchmark while
        # preserving fp16 payloads in the CUDA benchmark.
        dtype = (
            torch.float16
            if device.type == "cuda"
            and getattr(value, "dtype", None) == np.float16
            else torch.float32
        )
        model_input.append(torch.as_tensor(value, dtype=dtype, device=device))
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    return model_input, target, int(len(x))


def _profile_supported_flops(
    engine: Any,
    model_input: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Return profiler FLOPs for supported ops; this is deliberately labelled lower-bound."""
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        _sync(device)
        with profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with torch.inference_mode():
                engine._forward(model_input)
            _sync(device)
        flops = int(sum(int(getattr(event, "flops", 0) or 0) for event in prof.key_averages()))
        return {
            "status": "ok" if flops > 0 else "no_supported_flops_reported",
            "scope": "one_batch_supported_torch_ops_lower_bound",
            "batch_size": int(batch_size),
            "flops_per_batch_lower_bound": flops,
            "flops_per_sample_lower_bound": float(flops / max(batch_size, 1)),
            "macs_per_sample_approx_lower_bound": float(flops / max(2 * batch_size, 1)),
            "warning": (
                "torch.profiler counts FLOPs only for supported operators; FFT, "
                "indexing, interpolation and custom operations may be omitted."
            ),
        }
    except Exception as exc:  # profiling support varies across torch builds
        return {"status": "failed", "error": repr(exc)}


def _benchmark_calibrator(
    prediction_archive: Path,
    calibrator_path: Path,
    repeats: int,
) -> dict[str, Any]:
    from calibrate_dualscale_smooth_tod import (
        smooth_correction_for,
        target_time_correction_for,
    )

    with np.load(prediction_archive, allow_pickle=False) as archive:
        prediction = np.asarray(archive["test_prediction"], dtype=np.float64)
        endpoints = np.asarray(archive["test_endpoints"], dtype=np.int64)
    with np.load(calibrator_path, allow_pickle=False) as archive:
        centers = np.asarray(archive["centers"], dtype=np.float64)
        residuals = np.asarray(archive["smooth_residuals"], dtype=np.float64)
        time_table = np.asarray(
            archive["target_time_residuals"], dtype=np.float64
        )
        smooth_shrinkage = float(archive["smooth_shrinkage"])
        time_shrinkage = float(archive["time_shrinkage"])
        slices_per_day = int(archive["slices_per_day"])

    def apply_once() -> np.ndarray:
        smooth = prediction + smooth_shrinkage * smooth_correction_for(
            prediction, centers, residuals
        )
        return smooth + time_shrinkage * target_time_correction_for(
            endpoints,
            prediction.shape[1],
            time_table,
            slices_per_day,
        )

    # The implementation is deliberately the exact frozen implementation used
    # by the experiment, rather than an optimized replacement.
    apply_once()
    times = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        output = apply_once()
        times.append(time.perf_counter() - start)
        checksum = float(output.sum(dtype=np.float64))
    timing = _percentiles(times)
    timing["mean_ms_per_sample"] = float(
        timing["mean"] * 1000.0 / len(prediction)
    )
    return {
        "prediction_archive": str(prediction_archive),
        "calibrator": str(calibrator_path),
        "samples": int(len(prediction)),
        "prediction_shape": list(prediction.shape),
        "repeats": int(repeats),
        "cpu_seconds_per_full_archive": timing,
        "checksum": checksum,
        "warning": (
            "Calibration latency is measured separately from model inference. "
            "Add it to CPU end-to-end latency only when both use the same sample scope."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one PEMS04 checkpoint under a HyperD-aligned protocol."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--calibration-predictions", type=Path)
    parser.add_argument("--calibration-repeats", type=int, default=5)
    parser.add_argument(
        "--retrieval-cache-override",
        type=Path,
        help=(
            "Optional read-only retrieval-cache replacement used to audit "
            "storage/quantization variants without changing the checkpoint "
            "or its saved experiment config."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.warmup_batches < 0 or args.repeats <= 0:
        raise ValueError("batch-size/repeats must be positive and warmup non-negative")
    if (args.calibrator is None) != (args.calibration_predictions is None):
        raise ValueError(
            "--calibrator and --calibration-predictions must be passed together"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")

    exp_dir = args.exp_dir.resolve()
    config_path = exp_dir / "config.pkl"
    checkpoint_path = exp_dir / "best.pth"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Expected config.pkl and best.pth under {exp_dir}"
        )

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    logger = _logger()

    with config_path.open("rb") as stream:
        model_args = pickle.load(stream)
    if args.retrieval_cache_override is not None:
        override_path = args.retrieval_cache_override.resolve()
        if not override_path.is_file():
            raise FileNotFoundError(
                f"Retrieval-cache override not found: {override_path}"
            )
        model_args.analog_retrieval_cache = str(override_path)
    model_args.device = str(device)
    model_args.batch_size = int(args.batch_size)
    model_args.cache_data_path = "datasets"
    model_args.pearson_adj = None
    if isinstance(getattr(model_args, "k_hops", "1,2"), str):
        model_args.k_hops = [int(v) for v in model_args.k_hops.split(",")]

    # run.py owns a top-level argparse parser.  Strip this script's arguments
    # only during import, then restore them for honest provenance reporting.
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    from run import trainer, unpack_dataloader_input
    from utils import datasets_utils
    sys.argv = saved_argv

    process = __import__("psutil").Process()
    rss_after_import = int(process.memory_info().rss)
    dataloader, num_nodes, scaler, num_test_samples = datasets_utils.load_dataset(
        model_args, logger
    )
    model_args.num_nodes = num_nodes
    rss_after_data = int(process.memory_info().rss)
    engine = trainer(scaler=scaler, device=device, args=model_args, inverse=True)
    state = torch.load(checkpoint_path, map_location=device)
    engine.model.load_state_dict(state, strict=True)
    del state
    engine.model.eval()
    rss_after_model = int(process.memory_info().rss)

    total_params = int(sum(p.numel() for p in engine.model.parameters()))
    trainable_params = int(
        sum(p.numel() for p in engine.model.parameters() if p.requires_grad)
    )
    parameter_bytes = int(
        sum(p.numel() * p.element_size() for p in engine.model.parameters())
    )

    loader = dataloader[f"{args.split}_loader"]
    if args.split == "test":
        # test_loader pads the final batch by repeating its last endpoint.
        # The official unique sample count is returned separately by
        # load_dataset and must be used for metrics and reporting.
        logical_samples = int(num_test_samples)
    else:
        loader_size = getattr(loader, "size", None)
        if loader_size is None:
            loader_indices = getattr(loader, "idx", None)
            if loader_indices is None:
                raise AttributeError("Loader exposes neither size nor idx")
            loader_size = len(loader_indices)
        logical_samples = int(loader_size)
    first_batch = next(iter(loader.get_iterator()))
    warm_input, _, warm_batch_size = _prepare_batch(
        first_batch, device, unpack_dataloader_input
    )
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            engine._forward(warm_input)
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loader_times: list[float] = []
    transfer_times: list[float] = []
    forward_times: list[float] = []
    end_to_end_times: list[float] = []
    per_sample_forward_ms: list[float] = []
    per_sample_end_to_end_ms: list[float] = []
    peak_rss = int(process.memory_info().rss)
    total_abs = 0.0
    total_sq = 0.0
    total_valid = 0
    first_trial_seen = 0
    processed_samples_all_trials = 0

    for repeat in range(args.repeats):
        iterator = iter(loader.get_iterator())
        while True:
            batch_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            loaded = time.perf_counter()
            model_input, target, physical_batch = _prepare_batch(
                batch, device, unpack_dataloader_input
            )
            _sync(device)
            transferred = time.perf_counter()
            with torch.inference_mode():
                prediction = engine._forward(model_input)
            _sync(device)
            forwarded = time.perf_counter()

            loader_s = loaded - batch_start
            transfer_s = transferred - loaded
            forward_s = forwarded - transferred
            end_to_end_s = forwarded - batch_start
            loader_times.append(loader_s)
            transfer_times.append(transfer_s)
            forward_times.append(forward_s)
            end_to_end_times.append(end_to_end_s)
            per_sample_forward_ms.append(1000.0 * forward_s / physical_batch)
            per_sample_end_to_end_ms.append(
                1000.0 * end_to_end_s / physical_batch
            )
            processed_samples_all_trials += physical_batch

            if repeat == 0 and first_trial_seen < logical_samples:
                keep = min(physical_batch, logical_samples - first_trial_seen)
                pred_keep = prediction[:keep]
                target_keep = target[:keep]
                valid = torch.isfinite(target_keep) & (target_keep != 0)
                errors = (pred_keep - target_keep)[valid]
                total_abs += float(errors.abs().sum().cpu())
                total_sq += float(errors.square().sum().cpu())
                total_valid += int(errors.numel())
                first_trial_seen += keep
            peak_rss = max(peak_rss, int(process.memory_info().rss))

    if first_trial_seen != logical_samples or total_valid == 0:
        raise RuntimeError(
            f"Metric scope mismatch: seen={first_trial_seen}, "
            f"expected={logical_samples}, valid={total_valid}"
        )

    storage_records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def add_storage(path: Path | None, role: str) -> None:
        if path is None or path in seen_paths:
            return
        seen_paths.add(path)
        record = _file_record(path, role)
        if record is not None:
            storage_records.append(record)

    add_storage(checkpoint_path, "fp32_checkpoint_used")
    fp16_path = exp_dir / "best_fp16.pth"
    if fp16_path.is_file():
        add_storage(fp16_path, "optional_fp16_checkpoint")
    if bool(getattr(model_args, "use_analog_lowrank_retrieval_residual", 0)) or bool(
        getattr(model_args, "use_analog_exact_match_residual", 0)
    ):
        add_storage(
            _resolve_path(getattr(model_args, "analog_retrieval_cache", None)),
            "retrieval_cache",
        )
    if bool(getattr(model_args, "use_analog_candidate_reranker", 0)):
        add_storage(
            _resolve_path(getattr(model_args, "analog_candidate_cache", None)),
            "candidate_cache",
        )
    calibrator_path = (
        args.calibrator.resolve() if args.calibrator is not None else None
    )
    add_storage(calibrator_path, "calibrator")

    existing_bytes = [
        int(record["bytes"])
        for record in storage_records
        if record.get("exists") and record["role"] != "optional_fp16_checkpoint"
    ]
    fp16_checkpoint_bytes = next(
        (
            int(record["bytes"])
            for record in storage_records
            if record["role"] == "optional_fp16_checkpoint" and record.get("exists")
        ),
        None,
    )
    fp32_checkpoint_bytes = int(checkpoint_path.stat().st_size)
    external_bytes = int(sum(existing_bytes) - fp32_checkpoint_bytes)

    result: dict[str, Any] = {
        "protocol": "hyperd_aligned_pems04_efficiency_v1",
        "name": args.name,
        "experiment_directory": str(exp_dir),
        "retrieval_cache_override": (
            str(args.retrieval_cache_override.resolve())
            if args.retrieval_cache_override is not None
            else None
        ),
        "split": args.split,
        "logical_samples": logical_samples,
        "batch_size": int(args.batch_size),
        "warmup_batches": int(args.warmup_batches),
        "repeats": int(args.repeats),
        "device": {
            "requested": args.device,
            "torch_device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cpu": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_threads": int(args.cpu_threads),
        },
        "model": {
            "parameters_total": total_params,
            "parameters_trainable_flag_at_load": trainable_params,
            "parameter_tensor_bytes_at_runtime": parameter_bytes,
            "parameter_tensor_mib_at_runtime": parameter_bytes / (1024**2),
        },
        "accuracy_first_trial": {
            "mae_global_valid_elements": total_abs / total_valid,
            "rmse_global_valid_elements": float(np.sqrt(total_sq / total_valid)),
            "valid_elements": total_valid,
            "warning": (
                "This is a benchmark sanity metric. Official paper metrics may "
                "use horizon-wise aggregation; do not silently mix the two."
            ),
        },
        "timing": {
            "loader_seconds_per_batch": _percentiles(loader_times),
            "host_to_device_seconds_per_batch": _percentiles(transfer_times),
            "forward_seconds_per_batch": _percentiles(forward_times),
            "end_to_end_seconds_per_batch": _percentiles(end_to_end_times),
            "forward_ms_per_physical_sample": _percentiles(
                per_sample_forward_ms
            ),
            "end_to_end_ms_per_physical_sample": _percentiles(
                per_sample_end_to_end_ms
            ),
            "forward_throughput_samples_per_second": float(
                processed_samples_all_trials / max(sum(forward_times), 1e-12)
            ),
            "end_to_end_throughput_samples_per_second": float(
                processed_samples_all_trials / max(sum(end_to_end_times), 1e-12)
            ),
            "physical_samples_all_trials": int(processed_samples_all_trials),
        },
        "memory": {
            "process_rss_after_import_mib": rss_after_import / (1024**2),
            "process_rss_after_dataset_mib": rss_after_data / (1024**2),
            "process_rss_after_model_mib": rss_after_model / (1024**2),
            "process_peak_rss_observed_mib": peak_rss / (1024**2),
        },
        "storage": {
            "files": storage_records,
            "fp32_complete_deployment_bytes": int(sum(existing_bytes)),
            "fp32_complete_deployment_mib": float(
                sum(existing_bytes) / (1024**2)
            ),
            "external_assets_bytes": external_bytes,
            "external_assets_mib": external_bytes / (1024**2),
            "fp16_complete_deployment_bytes_if_available": (
                external_bytes + fp16_checkpoint_bytes
                if fp16_checkpoint_bytes is not None
                else None
            ),
            "fp16_complete_deployment_mib_if_available": (
                (external_bytes + fp16_checkpoint_bytes) / (1024**2)
                if fp16_checkpoint_bytes is not None
                else None
            ),
            "warning": (
                "Complete deployment storage includes only files discovered "
                "from this experiment config plus the explicitly supplied calibrator."
            ),
        },
    }
    if device.type == "cuda":
        result["memory"].update(
            {
                "peak_gpu_allocated_mib": float(
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                ),
                "peak_gpu_reserved_mib": float(
                    torch.cuda.max_memory_reserved(device) / (1024**2)
                ),
            }
        )
    if args.profile_flops:
        result["flops"] = _profile_supported_flops(
            engine, warm_input, device, warm_batch_size
        )
    else:
        result["flops"] = {"status": "not_requested"}

    if calibrator_path is not None:
        result["calibration_benchmark"] = _benchmark_calibrator(
            args.calibration_predictions.resolve(),
            calibrator_path,
            args.calibration_repeats,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
