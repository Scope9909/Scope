"""Seed-1 Validation forward with sparse retrieval tensors kept sparse in RAM/GPU.

This evaluator supports audited PEMS04/PEMS07/PEMS08 checkpoints with only low-rank
and exact-history retrieval adapters enabled. It never materializes [B,H,N,C]
retrieval/exact tensors; only the backbone prediction has all nodes.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=Path, required=True)
    p.add_argument("--sparse-cache", type=Path, required=True)
    p.add_argument("--no-memory", type=Path)
    p.add_argument("--full-memory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--predictions-output", type=Path,
                   help="Optional Validation predictions for chronological calibration")
    p.add_argument("--algebraic-reference-predictions", type=Path,
                   help="Same-evaluator mixed prediction archive for grouped memory parity")
    p.add_argument("--skip-reference", action="store_true",
                   help="Export native predictions without cross-archive parity comparison")
    p.add_argument("--disable-memory", action="store_true",
                   help="Backbone-only control with the same frozen checkpoint")
    p.add_argument("--max-batches", type=int, default=0,
                   help="0 means all Validation; positive value is a smoke check")
    return p.parse_args()


CLI = arguments()
sys.argv = [sys.argv[0]]  # run.py has a legacy global argument parser.
from run import trainer, unpack_dataloader_input  # noqa: E402
from utils import datasets_utils  # noqa: E402
from calibrate_dualscale_quantile_residual import metrics  # noqa: E402


def cpu_rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss
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
        get_memory.restype = wintypes.BOOL
        handle = get_process()
        if get_memory(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return None


def sparse_correction(model, prediction, ids, retrieval, exact, available):
    """Apply the checkpoint's low-rank then exact adapters at selected IDs."""
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


def main():
    if not CLI.skip_reference and (CLI.no_memory is None or CLI.full_memory is None):
        raise ValueError("--no-memory and --full-memory are required unless --skip-reference")
    if CLI.output.exists():
        raise FileExistsError(f"Refusing to overwrite {CLI.output}")
    if CLI.predictions_output is not None and CLI.predictions_output.exists():
        raise FileExistsError(f"Refusing to overwrite {CLI.predictions_output}")
    with (CLI.exp_dir / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    if args.dataset_name not in {"PEMS04", "PEMS07", "PEMS08"}:
        raise ValueError("This native evaluator supports PEMS04/PEMS07/PEMS08 only")
    enabled = [k for k, v in vars(args).items() if k.startswith("use_analog") and bool(v)]
    if set(enabled) != {"use_analog_lowrank_retrieval_residual",
                        "use_analog_exact_match_residual"}:
        raise ValueError(f"Unsupported retrieval modules: {enabled}")
    if getattr(args, "use_graph_conv", 0):
        raise ValueError("Unsupported graph convolution configuration")
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.pearson_adj = None
    # Deployment-style Validation loader: no Teacher or dense retrieval channels.
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    log = logging.getLogger("native_sparse")
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)
    load_start = time.perf_counter()
    loaders, nodes, scaler, _ = datasets_utils.load_dataset(args, log)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(CLI.exp_dir / "best.pth",
                                            map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    # Keep learned adapter parameters, but let the backbone run without retrieval.
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False
    loader = loaders["val_loader"]
    sample_count = len(loader.idx)
    groups = []
    with np.load(CLI.sparse_cache, allow_pickle=False) as archive:
        grouped = "heterogeneous_memory_format" in archive.files
        if grouped:
            if str(np.asarray(archive["heterogeneous_memory_format"]).item()) != "grouped_v1":
                raise ValueError("Unsupported grouped memory format")
            archive_val_endpoints = np.asarray(archive["idx_val"])
            for mode in ("int8", "fp16", "fp32"):
                node_ids = np.asarray(archive[f"node_ids_{mode}"], dtype=np.int64)
                if len(node_ids) == 0:
                    continue
                groups.append((mode, node_ids,
                               np.asarray(archive[f"retrieval_val_{mode}"]),
                               np.asarray(archive[f"exact_retrieval_val_{mode}"]),
                               np.asarray(archive[f"exact_available_val_{mode}"]),
                               np.asarray(archive["retrieval_quantization_scale_int8"], dtype=np.float32)
                               if mode == "int8" else None))
        else:
            archive_val_endpoints = np.asarray(archive["idx_val"]) if "idx_val" in archive.files else None
            node_ids = (np.asarray(archive["selected_node_ids"], dtype=np.int64)
                        if "selected_node_ids" in archive.files
                        else np.arange(nodes, dtype=np.int64))
            is_int8 = "retrieval_quantization" in archive.files
            if is_int8:
                if str(np.asarray(archive["retrieval_quantization"]).item()) != "symmetric_int8_per_horizon_node":
                    raise ValueError("Unsupported retrieval quantization")
                if str(np.asarray(archive["retrieval_quantization_fit_split"]).item()) != "train_only":
                    raise ValueError("Quantization must be fitted on train only")
            groups.append(("int8" if is_int8 else "float", node_ids,
                           np.asarray(archive["retrieval_val"]),
                           np.asarray(archive["exact_retrieval_val"]),
                           np.asarray(archive["exact_available_val"]),
                           np.asarray(archive["retrieval_quantization_scale"], dtype=np.float32)
                           if is_int8 else None))
    ids_np = np.concatenate([group[1] for group in groups])
    if (len(ids_np) == 0 or len(np.unique(ids_np)) != len(ids_np) or
            np.any((ids_np < 0) | (ids_np >= nodes))):
        raise ValueError("Invalid or overlapping sparse node IDs")
    for mode, node_ids, retrieval, exact, available, quant_scale in groups:
        shape = (sample_count, args.predict_length, len(node_ids), 1)
        if any(a.shape != shape for a in (retrieval, exact, available)):
            raise ValueError(f"Inconsistent Validation arrays in {mode}")
        if mode == "int8":
            if (retrieval.dtype != np.int8 or
                    quant_scale.shape != (args.predict_length, len(node_ids), 1) or
                    not np.isfinite(quant_scale).all() or np.any(quant_scale <= 0)):
                raise ValueError("Invalid INT8 retrieval or scale")
        elif retrieval.dtype not in (np.float16, np.float32):
            raise ValueError("Unsupported retrieval dtype")
        if exact.dtype not in (np.float16, np.float32):
            raise ValueError("Unsupported exact retrieval dtype")
    device_groups = [(mode, torch.as_tensor(node_ids, dtype=torch.long, device=device),
                      retrieval, exact, available, quant_scale)
                     for mode, node_ids, retrieval, exact, available, quant_scale in groups]
    load_seconds = time.perf_counter() - load_start
    rss_after_load = cpu_rss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    old_drop, old_batches = loader.drop_last, loader.num_batch
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    predictions, truths, cursor, forward_seconds, batches = [], [], 0, 0.0, 0
    rss_peak = rss_after_load or 0
    try:
        with torch.inference_mode():
            for loader_input, loader_target in loader.get_iterator():
                if CLI.max_batches and batches >= CLI.max_batches:
                    break
                x, x_time, *extras = unpack_dataloader_input(loader_input)
                if any(value is not None for value in extras):
                    raise ValueError("Validation loader unexpectedly includes retrieval channels")
                y, y_time = loader_target
                batch = len(x)
                model_input = [
                    torch.as_tensor(x, dtype=torch.float32, device=device),
                    torch.as_tensor(x_time, dtype=torch.long, device=device),
                    torch.as_tensor(y_time, dtype=torch.long, device=device),
                ]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tic = time.perf_counter()
                prediction, _ = model(model_input)
                if not CLI.disable_memory:
                    for mode, ids, retrieval, exact, available, quant_scale in device_groups:
                        retr = torch.as_tensor(retrieval[cursor:cursor + batch], device=device).float()
                        if mode == "int8":
                            retr = retr * torch.as_tensor(quant_scale, device=device)[None, ...]
                        ex = torch.as_tensor(exact[cursor:cursor + batch], device=device).float()
                        avail = torch.as_tensor(available[cursor:cursor + batch], device=device)
                        prediction = sparse_correction(model, prediction, ids, retr, ex, avail)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_seconds += time.perf_counter() - tic
                predictions.append(scaler.inverse_transform(prediction).cpu().numpy()[..., 0])
                truths.append(np.asarray(y)[..., 0])
                cursor += batch
                batches += 1
                rss_peak = max(rss_peak, cpu_rss() or 0)
    finally:
        loader.drop_last, loader.num_batch = old_drop, old_batches
    count = min(cursor, sample_count)
    prediction = np.concatenate(predictions)[:count]
    truth = np.concatenate(truths)[:count]
    index_path = (ROOT / "datasets" / args.dataset_name /
                  f"{args.dataset_name}_{args.input_length}to{args.predict_length}_index_scaler.npz")
    with np.load(index_path, allow_pickle=False) as archive:
        val_endpoints = np.asarray(archive["idx_val"][:count])
    if archive_val_endpoints is not None and not np.array_equal(
            archive_val_endpoints[:count], val_endpoints):
        raise ValueError("Sparse cache Validation endpoints differ from dataset")
    if CLI.skip_reference:
        algebraic = None
    elif CLI.algebraic_reference_predictions is not None:
        with np.load(CLI.algebraic_reference_predictions, allow_pickle=False) as archive:
            algebraic = np.asarray(archive["val_prediction"][:count])
            if (not np.array_equal(truth, archive["val_truth"][:count]) or
                    not np.array_equal(val_endpoints, archive["val_endpoints"][:count])):
                raise ValueError("Algebraic reference is unaligned")
    else:
        with np.load(CLI.no_memory, allow_pickle=False) as archive:
            no = np.asarray(archive["val_prediction"][:count])
            if not np.array_equal(truth, archive["val_truth"][:count]):
                raise ValueError("No-memory truth differs from dataset")
            if not np.array_equal(val_endpoints, archive["val_endpoints"][:count]):
                raise ValueError("No-memory Validation endpoints differ")
        with np.load(CLI.full_memory, allow_pickle=False) as archive:
            full = np.asarray(archive["val_prediction"][:count])
            if not np.array_equal(truth, archive["val_truth"][:count]):
                raise ValueError("Full-memory truth differs from dataset")
            if not np.array_equal(val_endpoints, archive["val_endpoints"][:count]):
                raise ValueError("Full-memory Validation endpoints differ")
        algebraic = no.copy()
        algebraic[:, :, ids_np] = full[:, :, ids_np]
    max_difference = (float(np.max(np.abs(prediction - algebraic)))
                      if algebraic is not None else None)
    mean_difference = (float(np.mean(np.abs(prediction - algebraic)))
                       if algebraic is not None else None)
    raw_mae = metrics(prediction, truth)["mae"]
    algebraic_mae = metrics(algebraic, truth)["mae"] if algebraic is not None else None
    report = {
        "protocol": "native_sparse_validation_forward_raw_uncalibrated",
        "dataset": args.dataset_name,
        "checkpoint": str(CLI.exp_dir.resolve() / "best.pth"),
        "test_evaluated": False, "smoke_only": bool(CLI.max_batches),
        "memory_disabled": CLI.disable_memory,
        "samples": count, "selected_nodes": len(ids_np),
        "memory_format": "grouped_v1" if grouped else "sparse_v1",
        "mode_counts": {mode: len(node_ids) for mode, node_ids, *_ in groups},
        "retrieval_stored_dtype": ",".join(str(group[2].dtype) for group in groups),
        "exact_stored_dtype": ",".join(str(group[3].dtype) for group in groups),
        "quantized_train_only": any(group[0] == "int8" for group in groups),
        "lossy_cache_reference_comparison":
            any(group[0] == "int8" or group[2].dtype == np.float16 or
                group[3].dtype == np.float16 for group in groups),
        "cache_bytes": CLI.sparse_cache.stat().st_size,
        "archive_load_and_model_init_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "forward_ms_per_sample": 1000 * forward_seconds / count,
        "cpu_rss_after_load_bytes": rss_after_load,
        "cpu_rss_peak_sampled_bytes": rss_peak or None,
        "gpu_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                     if device.type == "cuda" else None),
        "prediction_max_difference_vs_algebraic": max_difference,
        "prediction_mean_difference_vs_algebraic": mean_difference,
        "raw_mae": raw_mae,
        "algebraic_raw_mae": algebraic_mae,
        "passed_mean_prediction_and_mae_parity":
            (mean_difference <= 1e-5 and abs(raw_mae - algebraic_mae) <= 1e-4)
            if algebraic is not None else None,
    }
    print(json.dumps(report, indent=2))
    CLI.output.parent.mkdir(parents=True, exist_ok=True)
    CLI.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if CLI.predictions_output is not None:
        CLI.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CLI.predictions_output,
                            val_prediction=prediction.astype(np.float32),
                            val_truth=truth.astype(np.float32),
                            val_endpoints=val_endpoints)


if __name__ == "__main__":
    main()
