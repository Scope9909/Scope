"""Validation-only end-to-end benchmark with live Train-memory retrieval.

Modes are isolated per process: backbone-only, sparse online memory, or full-node
online memory. No precomputed Validation retrieval/query cache is accepted.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("backbone", "sparse", "full"), required=True)
    parser.add_argument("--state", type=Path,
                        help="Required for sparse/full; ignored for backbone")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--calibration", type=Path,
                        help="Existing fixed calibration protocol; Validation fields only")
    parser.add_argument("--reference-predictions", type=Path,
                        help="Optional aligned cache-based Validation predictions for parity")
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode != "backbone" and args.state is None:
        parser.error("--state is required for sparse/full")
    if args.batch_size < 1 or args.warmup_passes < 0 or args.repeats < 1:
        parser.error("Invalid batch size, warmup count, or repeat count")
    if args.output.exists() or (args.predictions_output and args.predictions_output.exists()):
        raise FileExistsError("Refusing to overwrite benchmark output")
    return args


CLI = parse_args()
sys.argv = [sys.argv[0]]
from run import trainer, unpack_dataloader_input  # noqa: E402
from utils import datasets_utils  # noqa: E402
from utils.online_residual_memory import OnlineResidualMemory  # noqa: E402
from calibrate_dualscale_quantile_residual import metrics  # noqa: E402
from calibrate_dualscale_smooth_tod import fit_and_apply  # noqa: E402


def cpu_rss() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        if sys.platform != "win32":
            return None
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if get_memory(get_process(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return None


class RssSampler:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak = cpu_rss() or 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, cpu_rss() or 0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, cpu_rss() or 0)


def sparse_correction(model, prediction, ids, retrieval, exact, available):
    low = model.analog_lowrank_retrieval_residual
    exact_adapter = model.analog_exact_match_residual
    selected = prediction.index_select(2, ids)
    logits = (low.horizon_logits[:, None] +
              torch.einsum("nr,hr->hn", low.node_factor.index_select(0, ids),
                           low.horizon_factor))
    blend = low.max_blend * torch.tanh(logits).view(1, logits.shape[0], len(ids), 1)
    selected = selected + blend * (retrieval - selected)
    exact_blend = (exact_adapter.max_blend *
                   torch.tanh(exact_adapter.horizon_logits)).view(1, -1, 1, 1)
    selected = selected + exact_blend * available.to(selected.dtype) * (exact - selected)
    return prediction.index_copy(2, ids, selected)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def validation_calibration(prediction, truth, endpoints, path: Path | None):
    if path is None:
        return None
    reference = json.loads(path.read_text(encoding="utf-8"))
    parameters = {key: reference[key] for key in (
        "num_bins", "smooth_shrinkage", "num_time_bins",
        "time_shrinkage", "slices_per_day")}
    rows = []
    for fold in reference["validation_folds"]:
        fit_end = int(fold["train_samples"])
        eval_start = int(fold.get("evaluation_start", fit_end))
        eval_end = int(fold["evaluation_end"])
        if fit_end != eval_start or eval_end > len(truth):
            raise ValueError("Calibration folds do not align with Validation")
        _, calibrated, _ = fit_and_apply(
            prediction[:fit_end], truth[:fit_end], endpoints[:fit_end],
            prediction[eval_start:eval_end], endpoints[eval_start:eval_end],
            **parameters)
        rows.append({
            "fit_end": fit_end, "eval_start": eval_start, "eval_end": eval_end,
            "raw_mae": metrics(prediction[eval_start:eval_end],
                               truth[eval_start:eval_end])["mae"],
            "fixed_calibrated_mae": metrics(calibrated,
                                            truth[eval_start:eval_end])["mae"],
        })
    return {
        "reference": str(path.resolve()),
        "folds": rows,
        "mean_raw_mae": float(np.mean([row["raw_mae"] for row in rows])),
        "mean_fixed_calibrated_mae": float(np.mean(
            [row["fixed_calibrated_mae"] for row in rows])),
    }


def main():
    load_start = time.perf_counter()
    with (CLI.exp_dir / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    if args.dataset_name not in {"PEMS03", "PEMS04", "PEMS07", "PEMS08"}:
        raise ValueError("Unsupported dataset")
    enabled = [key for key, value in vars(args).items()
               if key.startswith("use_analog") and bool(value)]
    if set(enabled) != {"use_analog_lowrank_retrieval_residual",
                        "use_analog_exact_match_residual"}:
        raise ValueError(f"Unsupported retrieval modules: {enabled}")
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = CLI.batch_size
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.pearson_adj = None
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("online_e2e")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.WARNING)
    loaders, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(
        CLI.exp_dir / "best.pth", map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False

    memory_state = None
    if CLI.mode != "backbone":
        with np.load(CLI.state, allow_pickle=False) as archive:
            memory_state = {key: archive[key] for key in archive.files}
    memory = None if memory_state is None else OnlineResidualMemory(memory_state)
    if memory is not None:
        if (memory.input_length != args.input_length or
                memory.predict_length != args.predict_length):
            raise ValueError("Online state horizon differs from the model")
        if CLI.mode == "full" and not np.array_equal(
                memory.node_ids, np.arange(nodes)):
            raise ValueError("Full mode requires an all-node state in dataset order")
        if CLI.mode == "sparse" and len(memory.node_ids) >= nodes:
            raise ValueError("Sparse mode requires a strict node subset")
        dataset_mean = np.asarray(scaler.mean).reshape(-1)
        dataset_std = np.asarray(scaler.std).reshape(-1)
        expected_mean = dataset_mean if dataset_mean.size == 1 else dataset_mean[memory.node_ids]
        expected_std = dataset_std if dataset_std.size == 1 else dataset_std[memory.node_ids]
        if not (np.array_equal(np.asarray(memory.scaler_mean).reshape(-1),
                               np.asarray(expected_mean).reshape(-1)) and
                np.array_equal(np.asarray(memory.scaler_std).reshape(-1),
                               np.asarray(expected_std).reshape(-1))):
            raise ValueError("Online state scaler differs from active dataset scaler")
    loader = loaders["val_loader"]
    index_path = (ROOT / "datasets" / args.dataset_name /
                  f"{args.dataset_name}_{args.input_length}to{args.predict_length}_index_scaler.npz")
    with np.load(index_path, allow_pickle=False) as archive:
        val_endpoints_all = np.asarray(archive["idx_val"], dtype=np.int64)
    with np.load(ROOT / "datasets" / args.dataset_name /
                 f"{args.dataset_name}_time.npz", allow_pickle=False) as archive:
        time_data = np.asarray(archive["time"])
    limit = len(val_endpoints_all)
    if CLI.max_validation_samples:
        limit = min(limit, CLI.max_validation_samples)
    val_endpoints = val_endpoints_all[:limit]
    if memory is not None:
        memory.validate_causality(val_endpoints,
                                  np.asarray(time_data[val_endpoints, 0], dtype=np.int16))
        if (memory.cold_start_endpoint is not None and
                memory.cold_start_endpoint != int(val_endpoints[0])):
            raise ValueError("Cold-start snapshot is not aligned to Validation start")
    ids = (None if memory is None else
           torch.as_tensor(memory.node_ids, dtype=torch.long, device=device))
    load_seconds = time.perf_counter() - load_start
    rss_after_load = cpu_rss()

    old_drop, old_batches = loader.drop_last, loader.num_batch
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))

    def run_pass(collect: bool):
        pass_memory = (OnlineResidualMemory(memory_state)
                       if memory is not None and memory.pending_count else memory)
        cursor = 0
        retrieval_seconds = 0.0
        model_seconds = 0.0
        loader_batch_seconds = []
        compute_batch_seconds = []
        end_to_end_batch_seconds = []
        predictions, truths = [], []
        wall_start = time.perf_counter()
        with torch.inference_mode():
            iterator = iter(loader.get_iterator())
            while True:
                loader_start = time.perf_counter()
                try:
                    loader_input, loader_target = next(iterator)
                except StopIteration:
                    break
                loader_elapsed = time.perf_counter() - loader_start
                if cursor >= limit:
                    break
                x, x_time, *extras = unpack_dataloader_input(loader_input)
                if any(value is not None for value in extras):
                    raise ValueError("Validation loader contains a precomputed query cache")
                y, y_time = loader_target
                take = min(len(x), limit - cursor)
                x, x_time, y, y_time = x[:take], x_time[:take], y[:take], y_time[:take]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                batch_start = time.perf_counter()
                if pass_memory is not None:
                    # PEMS traffic values are integer-valued.  Recover the exact
                    # observed input before nearest/exact-history matching; a plain
                    # float32 inverse transform leaves sub-ULP noise that can break
                    # equality matches and alter distance ties.
                    raw_history = np.rint(
                        scaler.inverse_transform(x)[..., 0][:, :, pass_memory.node_ids]
                    ).astype(np.float32)
                    query_tod = np.asarray(x_time[:, -1, 0], dtype=np.int16)
                    retrieval_start = time.perf_counter()
                    if pass_memory.cold_start_endpoint is None:
                        retrieval, exact, available, _ = pass_memory.query(
                            raw_history, query_tod)
                    else:
                        retrieval, exact, available, _ = pass_memory.query_stream(
                            raw_history, val_endpoints[cursor:cursor + take], query_tod)
                    retrieval_seconds += time.perf_counter() - retrieval_start
                model_start = time.perf_counter()
                model_input = [
                    torch.as_tensor(x, dtype=torch.float32, device=device),
                    torch.as_tensor(x_time, dtype=torch.long, device=device),
                    torch.as_tensor(y_time, dtype=torch.long, device=device),
                ]
                prediction, _ = model(model_input)
                if pass_memory is not None:
                    prediction = sparse_correction(
                        model, prediction, ids,
                        torch.as_tensor(retrieval, device=device),
                        torch.as_tensor(exact, device=device).float(),
                        torch.as_tensor(available, device=device))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_seconds += time.perf_counter() - model_start
                compute_elapsed = time.perf_counter() - batch_start
                loader_batch_seconds.append(loader_elapsed)
                compute_batch_seconds.append(compute_elapsed)
                end_to_end_batch_seconds.append(loader_elapsed + compute_elapsed)
                if collect:
                    predictions.append(scaler.inverse_transform(prediction).cpu().numpy()[..., 0])
                    truths.append(np.asarray(y)[..., 0])
                cursor += take
        wall_seconds = time.perf_counter() - wall_start
        result = {
            "samples": cursor,
            "wall_seconds": wall_seconds,
            "retrieval_seconds": retrieval_seconds,
            "model_and_transfer_seconds": model_seconds,
            "loader_batch_seconds": loader_batch_seconds,
            "compute_batch_seconds": compute_batch_seconds,
            "end_to_end_batch_seconds": end_to_end_batch_seconds,
        }
        if collect:
            result["prediction"] = np.concatenate(predictions)[:cursor]
            result["truth"] = np.concatenate(truths)[:cursor]
        return result

    try:
        for _ in range(CLI.warmup_passes):
            run_pass(False)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        measured = []
        first = None
        with RssSampler() as sampler:
            for repeat in range(CLI.repeats):
                current = run_pass(repeat == 0)
                if repeat == 0:
                    first = current
                measured.append({key: value for key, value in current.items()
                                 if key not in {"prediction", "truth"}})
        cpu_peak = sampler.peak
    finally:
        loader.drop_last, loader.num_batch = old_drop, old_batches

    prediction = first["prediction"]
    truth = first["truth"]
    endpoints = val_endpoints[:len(prediction)]
    raw_metrics = metrics(prediction, truth)
    loader_batches = [value for run in measured for value in run["loader_batch_seconds"]]
    compute_batches = [value for run in measured for value in run["compute_batch_seconds"]]
    end_to_end_batches = [value for run in measured
                          for value in run["end_to_end_batch_seconds"]]
    wall = [run["wall_seconds"] for run in measured]
    retrieval_time = [run["retrieval_seconds"] for run in measured]
    model_time = [run["model_and_transfer_seconds"] for run in measured]
    parity = None
    if CLI.reference_predictions is not None:
        with np.load(CLI.reference_predictions, allow_pickle=False) as archive:
            reference = np.asarray(archive["val_prediction"][:len(prediction)])
            if (not np.array_equal(truth, archive["val_truth"][:len(prediction)]) or
                    not np.array_equal(endpoints, archive["val_endpoints"][:len(prediction)])):
                raise ValueError("Reference predictions are not Validation-aligned")
        parity = {
            "reference": str(CLI.reference_predictions.resolve()),
            "max_absolute_difference": float(np.max(np.abs(prediction - reference))),
            "mean_absolute_difference": float(np.mean(np.abs(prediction - reference))),
        }
    report = {
        "protocol": "train_only_online_memory_end_to_end_validation_v1",
        "dataset": args.dataset_name,
        "mode": CLI.mode,
        "test_evaluated": False,
        "query_cache_loaded": False,
        "checkpoint": str((CLI.exp_dir / "best.pth").resolve()),
        "state": str(CLI.state.resolve()) if CLI.state else None,
        "state_bytes": CLI.state.stat().st_size if CLI.state else 0,
        "selected_nodes": len(memory.node_ids) if memory else 0,
        "retrieval_format": memory.retrieval_format if memory else None,
        "total_nodes": nodes,
        "cold_start_endpoint": memory.cold_start_endpoint if memory else None,
        "pending_candidates_at_cold_start": memory.pending_count if memory else 0,
        "source_train_candidates": memory.source_candidate_count if memory else 0,
        "device": str(device),
        "batch_size": CLI.batch_size,
        "warmup_passes": CLI.warmup_passes,
        "repeats": CLI.repeats,
        "samples_per_pass": len(prediction),
        "load_and_init_seconds": load_seconds,
        "pass_wall_seconds": wall,
        "pass_wall_median_seconds": float(np.median(wall)),
        "pass_wall_p95_seconds": percentile(wall, 95),
        "throughput_samples_per_second_at_median": float(len(prediction) / np.median(wall)),
        "retrieval_seconds": retrieval_time,
        "retrieval_median_seconds": float(np.median(retrieval_time)),
        "model_and_transfer_seconds": model_time,
        "model_and_transfer_median_seconds": float(np.median(model_time)),
        "loader_batch_latency_ms_p50": 1000 * percentile(loader_batches, 50),
        "loader_batch_latency_ms_p95": 1000 * percentile(loader_batches, 95),
        "compute_batch_latency_ms_p50": 1000 * percentile(compute_batches, 50),
        "compute_batch_latency_ms_p95": 1000 * percentile(compute_batches, 95),
        "end_to_end_batch_latency_ms_p50": 1000 * percentile(end_to_end_batches, 50),
        "end_to_end_batch_latency_ms_p95": 1000 * percentile(end_to_end_batches, 95),
        "batch_latency_scope": (
            f"one batch of at most {CLI.batch_size}; end-to-end includes loader assembly, "
            "online retrieval, host-to-device transfer, model forward, and sparse correction"),
        "cpu_rss_after_load_bytes": rss_after_load,
        "cpu_rss_peak_sampled_bytes": cpu_peak,
        "gpu_peak_allocated_bytes": (int(torch.cuda.max_memory_allocated(device))
                                     if device.type == "cuda" else None),
        "gpu_peak_reserved_bytes": (int(torch.cuda.max_memory_reserved(device))
                                    if device.type == "cuda" else None),
        "raw_validation_metrics": raw_metrics,
        "fixed_validation_calibration": validation_calibration(
            prediction, truth, endpoints, CLI.calibration),
        "prediction_parity": parity,
    }
    CLI.output.parent.mkdir(parents=True, exist_ok=True)
    CLI.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if CLI.predictions_output is not None:
        CLI.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CLI.predictions_output,
                            val_prediction=prediction.astype(np.float32),
                            val_truth=truth.astype(np.float32),
                            val_endpoints=endpoints.astype(np.int64))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
