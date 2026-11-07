"""Validation-only runtime benchmark for full-future versus code-domain candidates."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "code"), required=True)
    parser.add_argument(
        "--decode-candidates", type=int, choices=(0, 1), default=0,
        help="Code mode: decode K candidates for disagreement features (1) or not (0).",
    )
    parser.add_argument(
        "--exp-dir", type=Path,
        default=Path("EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3"),
    )
    parser.add_argument(
        "--code-cache", type=str,
        default="datasets/PEMS04/PEMS04_12to12_candidate_pca_r8_codes_val_fp16.npz",
    )
    parser.add_argument(
        "--candidate-cache", type=str, default=None,
        help=(
            "Override the train-only candidate endpoint cache while keeping "
            "the saved model checkpoint and reranker weights unchanged."
        ),
    )
    parser.add_argument(
        "--candidate-distance-metric", type=str, default=None,
        choices=(
            "masked_mse", "masked_mse_trend", "masked_mae",
            "masked_mae_trend_volatility",
        ),
        help="Distance metric metadata expected by the overridden candidate cache.",
    )
    parser.add_argument(
        "--retrieval-cache", type=str, default=None,
        help=(
            "Override the mean-retrieval/exact cache together with the "
            "candidate cache so both come from the same candidate rule."
        ),
    )
    parser.add_argument(
        "--candidate-selection-strategy", type=str, default=None,
        help=(
            "Selection strategy metadata expected by the overridden cache. "
            "Inferred from the cache when omitted. 'nearest', 'rank_spread' "
            "and 'mmr*' tags are supported."
        ),
    )
    parser.add_argument(
        "--reranker-uniform", action="store_true",
        help=(
            "Zero the frozen reranker's output projection after loading the "
            "checkpoint. Its logits become identically zero, the masked "
            "softmax degenerates to uniform and the prediction falls back to "
            "the candidate mean. Isolates whether a candidate-set change "
            "helps retrieval or is blocked by the reranker's training "
            "distribution."
        ),
    )
    parser.add_argument(
        "--level-align", type=int, choices=(0, 1), default=None,
        help="Override observable level-aligned candidate residual construction.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override the saved batch size so baseline/code use one protocol.",
    )
    parser.add_argument(
        "--split", choices=("val", "test"), default="val",
        help="Which split to evaluate. 'test' only replays a frozen model; "
             "it must never be used to select a configuration.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--slot-error-output", type=Path, default=None,
        help="Save per-target absolute errors and valid mask for paired audits.",
    )
    parser.add_argument(
        "--lost-exact-cache", type=Path, default=None,
        help="Restrict an auxiliary MAE to slots exact-available in the full cache but lost in this cache.",
    )
    parser.add_argument(
        "--full-exact-cache", type=Path, default=None,
        help="Full-memory reference retrieval cache for --lost-exact-cache.",
    )
    args = parser.parse_args()
    if (args.lost_exact_cache is None) != (args.full_exact_cache is None):
        parser.error("--lost-exact-cache and --full-exact-cache must be used together")
    import psutil

    with (args.exp_dir / "config.pkl").open("rb") as stream:
        model_args = pickle.load(stream)
    model_args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_args.cache_data_path = "datasets"
    model_args.pearson_adj = None
    if isinstance(getattr(model_args, "k_hops", "1,2"), str):
        model_args.k_hops = [int(v) for v in model_args.k_hops.split(",")]
    if args.mode == "code":
        model_args.use_analog_code_domain_aggregation = 1
        model_args.analog_candidate_code_cache = args.code_cache
        model_args.analog_code_domain_decode_candidates = args.decode_candidates
    else:
        model_args.use_analog_code_domain_aggregation = 0
        model_args.analog_candidate_code_cache = "none"
    if args.retrieval_cache is not None:
        model_args.analog_retrieval_cache = args.retrieval_cache
    if args.candidate_cache is not None:
        model_args.analog_candidate_cache = args.candidate_cache
        if args.candidate_distance_metric is not None:
            model_args.analog_candidate_distance_metric = (
                args.candidate_distance_metric
            )
        else:
            with np.load(args.candidate_cache, allow_pickle=False) as cache:
                if "history_distance_metric" not in cache:
                    raise ValueError(
                        "Overridden candidate cache is missing "
                        "history_distance_metric; pass --candidate-distance-metric"
                    )
                model_args.analog_candidate_distance_metric = str(
                    np.asarray(cache["history_distance_metric"]).item()
                )
        if args.candidate_selection_strategy is not None:
            model_args.analog_candidate_selection_strategy = (
                args.candidate_selection_strategy
            )
        else:
            with np.load(args.candidate_cache, allow_pickle=False) as cache:
                if "candidate_selection_strategy" not in cache:
                    raise ValueError(
                        "Overridden candidate cache is missing "
                        "candidate_selection_strategy; pass "
                        "--candidate-selection-strategy"
                    )
                model_args.analog_candidate_selection_strategy = str(
                    np.asarray(cache["candidate_selection_strategy"]).item()
                )
    if args.level_align is not None:
        model_args.analog_candidate_level_align = int(args.level_align)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        model_args.batch_size = int(args.batch_size)

    sys.argv = [sys.argv[0]]
    from run import trainer, unpack_dataloader_input
    from utils import datasets_utils

    logger = __import__("logging").getLogger("runtime_benchmark")
    logger.addHandler(__import__("logging").StreamHandler(sys.stdout))
    logger.setLevel(__import__("logging").INFO)
    dataloader, num_nodes, scaler, _ = datasets_utils.load_dataset(model_args, logger)
    model_args.num_nodes = num_nodes
    device = torch.device(model_args.device)
    engine = trainer(scaler=scaler, device=device, args=model_args, inverse=True)
    engine.model.load_state_dict(
        torch.load(args.exp_dir / "best.pth", map_location=device), strict=True
    )
    engine.model.eval()
    if args.reranker_uniform:
        reranker = getattr(engine.model, "analog_candidate_reranker", None)
        if reranker is None:
            raise ValueError(
                "reranker-uniform needs a checkpoint with the candidate reranker"
            )
        with torch.no_grad():
            reranker.output_projection.weight.zero_()
            reranker.output_projection.bias.zero_()
        print(
            "reranker-uniform: output projection zeroed, candidate weights "
            "are exactly uniform",
            flush=True,
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    loader_times, model_times, decode_times, aggregate_times = [], [], [], []
    final_decode_times, candidate_decode_counts, final_decode_counts = [], [], []
    batch_mae = []
    total_abs = total_sq = 0.0
    total_count = 0
    slot_errors = []
    slot_valid = []
    lost_exact_mask = None
    lost_abs = 0.0
    lost_count = 0
    query_offset = 0
    if args.lost_exact_cache is not None:
        with np.load(args.full_exact_cache, allow_pickle=False) as full_cache:
            full_available = full_cache["exact_available_val"].astype(bool)
        with np.load(args.lost_exact_cache, allow_pickle=False) as pruned_cache:
            pruned_available = pruned_cache["exact_available_val"].astype(bool)
        if full_available.shape != pruned_available.shape:
            raise ValueError("Full and pruned exact masks have different shapes")
        lost_exact_mask = full_available & ~pruned_available
    first_payload_shape = None
    loader_key = f"{args.split}_loader"
    iterator = iter(dataloader[loader_key].get_iterator())
    while True:
        t0 = time.perf_counter()
        try:
            loader_input, loader_target = next(iterator)
        except StopIteration:
            break
        t1 = time.perf_counter()
        (
            x, x_time, analog_retrieval,
            exact_retrieval, exact_available,
            candidate_future, candidate_available, candidate_features,
        ) = unpack_dataloader_input(loader_input)
        y, y_time = loader_target
        model_input = [
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(x_time, dtype=torch.long, device=device),
            torch.as_tensor(y_time, dtype=torch.long, device=device),
        ]
        for value in (
            analog_retrieval, exact_retrieval, exact_available,
            candidate_future, candidate_available, candidate_features,
        ):
            if value is not None:
                dtype = torch.float16 if getattr(value, "dtype", None) == np.float16 else torch.float32
                model_input.append(torch.as_tensor(value, dtype=dtype, device=device))
        if first_payload_shape is None and candidate_future is not None:
            first_payload_shape = list(candidate_future.shape)
        with torch.no_grad():
            t2 = time.perf_counter()
            prediction = engine._forward(model_input)
            t3 = time.perf_counter()
        target = torch.as_tensor(y, dtype=torch.float32, device=device)
        # Test loader pads its last batch by repeating the final query.  The
        # official run.py evaluation removes those copies before computing
        # metrics, so apply the same limit here.
        remaining = len(dataloader[loader_key].idx) - query_offset
        keep = min(len(y), remaining)
        if keep <= 0:
            break
        prediction = prediction[:keep]
        target = target[:keep]
        error = prediction - target
        valid = torch.isfinite(target) & (target != 0)
        valid_error = error[valid]
        if args.slot_error_output is not None:
            slot_errors.append(error.abs().cpu().numpy().astype(np.float32))
            slot_valid.append(valid.cpu().numpy())
        if lost_exact_mask is not None:
            batch_lost = torch.as_tensor(
                lost_exact_mask[query_offset:query_offset + keep],
                dtype=torch.bool, device=device,
            )
            restricted = valid & batch_lost
            lost_abs += float(error[restricted].abs().sum().cpu())
            lost_count += int(restricted.sum().cpu())
        query_offset += keep
        total_abs += float(valid_error.abs().sum().cpu())
        total_sq += float(valid_error.square().sum().cpu())
        total_count += int(valid_error.numel())
        batch_mae.append(float(valid_error.abs().mean().cpu()) if valid_error.numel() else 0.0)
        loader_times.append(t1 - t0)
        model_times.append(t3 - t2)
        if hasattr(engine.model, 'analog_candidate_reranker'):
            reranker = engine.model.analog_candidate_reranker
            decode_times.append(float(getattr(reranker, 'last_code_decode_seconds', 0.0)))
            aggregate_times.append(float(getattr(reranker, 'last_code_aggregate_seconds', 0.0)))
            final_decode_times.append(float(getattr(
                reranker, 'last_code_final_decode_seconds', 0.0
            )))
            candidate_decode_counts.append(int(getattr(
                reranker, 'last_code_candidate_decode_count', 0
            )))
            final_decode_counts.append(int(getattr(
                reranker, 'last_code_final_decode_count', 0
            )))
        peak_rss = max(peak_rss, process.memory_info().rss)

    if not total_count:
        raise RuntimeError("Validation produced no valid target elements")
    result = {
        "scope": f"{args.exp_dir.parent.name}_{args.split}_only_no_training",
        "mode": args.mode,
        "split": args.split,
        "reranker": "uniform" if args.reranker_uniform else "as-trained",
        "query_count": int(len(dataloader[loader_key].idx)),
        "candidate_payload_shape_first_batch": first_payload_shape,
        "metrics": {
            "mae": total_abs / total_count,
            "rmse": float(np.sqrt(total_sq / total_count)),
        },
        "timing_seconds": {
            "retrieval_loader_total": float(sum(loader_times)),
            "retrieval_loader_mean_batch": float(np.mean(loader_times)),
            "model_end_to_end_total": float(sum(model_times)),
            "model_end_to_end_mean_batch": float(np.mean(model_times)),
            "model_end_to_end_p50_batch": float(np.percentile(model_times, 50)),
            "model_end_to_end_p95_batch": float(np.percentile(model_times, 95)),
            "code_decode_total": float(sum(decode_times)),
            "code_decode_mean_batch": float(np.mean(decode_times)) if decode_times else 0.0,
            "code_aggregate_total": float(sum(aggregate_times)),
            "code_aggregate_mean_batch": float(np.mean(aggregate_times)) if aggregate_times else 0.0,
            "final_decode_total": float(sum(final_decode_times)),
            "final_decode_mean_batch": float(np.mean(final_decode_times)) if final_decode_times else 0.0,
            "candidate_decode_count_total": int(sum(candidate_decode_counts)),
            "final_decode_count_total": int(sum(final_decode_counts)),
        },
        "peak_process_rss_bytes": int(peak_rss),
        "peak_process_rss_mib": float(peak_rss / (1024 ** 2)),
    }
    if lost_exact_mask is not None:
        result["lost_exact_restricted"] = {
            "valid_slots": lost_count,
            "mae": lost_abs / lost_count if lost_count else None,
        }
    if torch.cuda.is_available():
        result["peak_gpu_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        result["peak_gpu_allocated_mib"] = float(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        result["peak_gpu_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
        result["peak_gpu_reserved_mib"] = float(
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.slot_error_output is not None:
        args.slot_error_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.slot_error_output,
            absolute_error=np.concatenate(slot_errors, axis=0),
            valid=np.concatenate(slot_valid, axis=0),
        )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
