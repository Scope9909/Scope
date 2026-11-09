"""Validation-only prototype of sensor-utility-budgeted residual memory.

Select sensors on the first 55% of Validation using paired error reduction
from the same frozen model. Evaluate only on the later 45% of Validation.
Sparse NPZ artifacts store retrieval/exact values for selected sensors only;
the decoder restores dense tensors in RAM, and a masked low-rank adapter
ensures absent sensors receive exactly zero retrieval correction.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, type=Path)
    parser.add_argument("--full-cache", required=True, type=Path)
    parser.add_argument("--full-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--save-no-memory-only", action="store_true")
    parser.add_argument("--sparse-cache", type=Path)
    parser.add_argument("--strict-teacher-cache", type=Path)
    return parser.parse_args()


CLI_ARGS = parse_args()
sys.argv = [sys.argv[0]]  # run.py owns a legacy global parser.

from run import trainer, unpack_dataloader_input  # noqa: E402
from utils import datasets_utils  # noqa: E402
from calibrate_dualscale_quantile_residual import metrics  # noqa: E402


def get_logger():
    log = logging.getLogger("node_budget_pilot")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
    return log


@torch.no_grad()
def collect(model, loader, scaler, device, sample_count):
    predictions, truths = [], []
    old_drop, old_batches = loader.drop_last, loader.num_batch
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    try:
        model.eval()
        for loader_input, loader_target in loader.get_iterator():
            (
                x, x_time, analog_retrieval, exact_retrieval,
                exact_available, candidate_future, candidate_available,
                candidate_features,
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
                    model_input.append(torch.as_tensor(value, device=device))
            prediction, _ = model(model_input)
            predictions.append(prediction.cpu())
            truths.append(torch.as_tensor(y, dtype=torch.float32))
    finally:
        loader.drop_last, loader.num_batch = old_drop, old_batches
    prediction = torch.cat(predictions, dim=0)[:sample_count]
    truth = torch.cat(truths, dim=0)[:sample_count]
    prediction = scaler.inverse_transform(prediction).numpy()[..., 0]
    truth = truth.numpy()[..., 0]
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Validation prediction/truth check failed")
    return prediction.astype(np.float32), truth.astype(np.float32)


def save_sparse_cache(source, destination, node_ids):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    arrays = {"selected_node_ids": node_ids.astype(np.int32)}
    with np.load(source, allow_pickle=False) as archive:
        for key in archive.files:
            value = archive[key]
            if key.startswith(("retrieval_", "exact_retrieval_", "exact_available_")) and key.endswith(("_train", "_val", "_test")):
                if value.ndim != 4:
                    raise ValueError(f"Expected four-dimensional {key}")
                arrays[key] = value[:, :, node_ids, :]
            else:
                arrays[key] = value
    np.savez_compressed(destination, **arrays)


def expand_validation_cache(path, sample_count, horizon, num_nodes):
    with np.load(path, allow_pickle=False) as archive:
        node_ids = np.asarray(archive["selected_node_ids"], dtype=np.int64)
        arrays = []
        for key, dtype in (
            ("retrieval_val", np.float32),
            ("exact_retrieval_val", np.float32),
            ("exact_available_val", np.uint8),
        ):
            sparse = np.asarray(archive[key])
            if sparse.shape != (sample_count, horizon, len(node_ids), 1):
                raise ValueError(f"Unexpected sparse {key} shape: {sparse.shape}")
            dense = np.zeros((sample_count, horizon, num_nodes, 1), dtype=dtype)
            dense[:, :, node_ids, :] = sparse
            arrays.append(dense)
    return node_ids, arrays


def masked_node_gain(no_memory, full_memory, truth, prefix_end):
    valid = np.round(truth[:prefix_end]) > 0
    denominator = valid.sum(axis=(0, 1))
    if np.any(denominator == 0):
        raise ValueError("A node has no valid target in selection prefix")
    no_error = (np.abs(no_memory[:prefix_end] - truth[:prefix_end]) * valid).sum(axis=(0, 1))
    full_error = (np.abs(full_memory[:prefix_end] - truth[:prefix_end]) * valid).sum(axis=(0, 1))
    return no_error / denominator - full_error / denominator


def main():
    exp_dir = CLI_ARGS.exp_dir.resolve()
    full_cache = CLI_ARGS.full_cache.resolve()
    output_dir = CLI_ARGS.output_dir.resolve()
    report_path = output_dir / "node_budget_validation.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    if CLI_ARGS.strict_teacher_cache is not None:
        args.teacher_predictions_path = str(CLI_ARGS.strict_teacher_cache.resolve())
        args.teacher_predictions_strict_alignment = 1
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.pearson_adj = None  # This exact configuration uses no graph convolution.
    if bool(getattr(args, "use_graph_conv", 0)):
        raise ValueError("This pilot supports the no-graph PEMS04 configuration only")
    lowrank_enabled = args.use_analog_lowrank_retrieval_residual
    exact_enabled = args.use_analog_exact_match_residual
    if not lowrank_enabled or not exact_enabled:
        raise ValueError("Expected low-rank and exact-memory adapters")
    # Load all ordinary dataset/KD channels without loading the full retrieval cache.
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    device = torch.device(args.device)
    log = get_logger()
    loaders, num_nodes, scaler, _ = datasets_utils.load_dataset(args, log)
    args.use_analog_lowrank_retrieval_residual = lowrank_enabled
    args.use_analog_exact_match_residual = exact_enabled
    args.num_nodes = num_nodes
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(
        torch.load(exp_dir / "best.pth", map_location=device, weights_only=True),
        strict=True,
    )
    loader = loaders["val_loader"]
    sample_count = len(loader.idx)
    horizon = args.predict_length
    with np.load(CLI_ARGS.full_predictions, allow_pickle=False) as archive:
        full_prediction = np.asarray(archive["val_prediction"], dtype=np.float32)
        full_truth = np.asarray(archive["val_truth"], dtype=np.float32)
        full_endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
    with np.load(Path("datasets") / args.dataset_name / (
        f"{args.dataset_name}_{args.input_length}to{horizon}_index_scaler.npz"
    ), allow_pickle=False) as archive:
        endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
    if not np.array_equal(endpoints, full_endpoints):
        raise ValueError("Full-memory validation endpoints do not match")
    if full_prediction.shape != (sample_count, horizon, num_nodes):
        raise ValueError("Full-memory prediction shape does not match")

    mask_holder = {"value": torch.zeros((1, 1, num_nodes, 1), device=device)}
    adapter = engine.model.analog_lowrank_retrieval_residual
    original_forward = adapter.forward

    def masked_forward(prediction, retrieval):
        correction = original_forward(prediction, retrieval) - prediction
        return prediction + mask_holder["value"] * correction

    adapter.forward = masked_forward
    loader.retrieval_data = np.zeros((sample_count, horizon, num_nodes, 1), dtype=np.float32)
    loader.exact_retrieval_data = np.zeros_like(loader.retrieval_data)
    loader.exact_available_data = np.zeros_like(loader.retrieval_data, dtype=np.uint8)
    log.info("Forwarding no-memory control on Validation ...")
    no_memory_prediction, truth = collect(
        engine.model, loader, scaler, device, sample_count
    )
    if not np.array_equal(truth, full_truth):
        raise ValueError("No-memory labels do not match full-memory labels")
    if CLI_ARGS.save_no_memory_only:
        np.savez_compressed(
            output_dir / "no_memory_validation_predictions.npz",
            val_prediction=no_memory_prediction,
            val_truth=truth,
            val_endpoints=endpoints,
        )
        log.info("Saved no-memory Validation predictions")
        return
    if CLI_ARGS.sparse_cache is not None:
        sparse_path = CLI_ARGS.sparse_cache.resolve()
        selected, arrays = expand_validation_cache(
            sparse_path, sample_count, horizon, num_nodes
        )
        loader.retrieval_data, loader.exact_retrieval_data, loader.exact_available_data = arrays
        node_mask = torch.zeros(num_nodes, device=device)
        node_mask[torch.as_tensor(selected, device=device)] = 1.0
        mask_holder["value"] = node_mask.view(1, 1, num_nodes, 1)
        log.info("Forwarding physical sparse cache with %d nodes ...", len(selected))
        sparse_prediction, sparse_truth = collect(
            engine.model, loader, scaler, device, sample_count
        )
        if not np.array_equal(sparse_truth, truth):
            raise ValueError("Sparse-cache labels differ")
        algebraic = no_memory_prediction.copy()
        algebraic[:, :, selected] = full_prediction[:, :, selected]
        difference = np.abs(sparse_prediction - algebraic)
        prefix_end = int(0.55 * sample_count)
        sparse_suffix = metrics(sparse_prediction[prefix_end:], truth[prefix_end:])
        algebraic_suffix = metrics(algebraic[prefix_end:], truth[prefix_end:])
        verification = {
            "test_evaluated": False,
            "selected_nodes": len(selected),
            "sparse_cache": str(sparse_path),
            "sparse_cache_bytes": sparse_path.stat().st_size,
            "sparse_cache_mib": sparse_path.stat().st_size / 2**20,
            "physical_sparse_suffix": sparse_suffix,
            "algebraic_sparse_suffix": algebraic_suffix,
            "physical_minus_algebraic_suffix_mae": sparse_suffix["mae"] - algebraic_suffix["mae"],
            "prediction_maximum_absolute_difference": float(difference.max()),
            "prediction_mean_absolute_difference": float(difference.mean()),
        }
        with (output_dir / "sparse_forward_verification.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(verification, stream, ensure_ascii=False, indent=2)
        log.info(
            "Physical sparse suffix MAE %.4f; algebraic difference %+.6f",
            sparse_suffix["mae"], verification["physical_minus_algebraic_suffix_mae"],
        )
        return
    if CLI_ARGS.verify_only:
        with np.load(full_cache, allow_pickle=False) as archive:
            loader.retrieval_data = np.asarray(archive["retrieval_val"], dtype=np.float32)
            loader.exact_retrieval_data = np.asarray(
                archive["exact_retrieval_val"], dtype=np.float32
            )
            loader.exact_available_data = np.asarray(
                archive["exact_available_val"], dtype=np.uint8
            )
        # Use the original unwrapped adapter to audit the evaluator itself.
        adapter.forward = original_forward
        reproduced, reproduced_truth = collect(
            engine.model, loader, scaler, device, sample_count
        )
        if not np.array_equal(reproduced_truth, truth):
            raise ValueError("Full-memory reproduced labels differ")
        absolute_difference = np.abs(reproduced - full_prediction)
        maximum_difference = float(np.max(absolute_difference))
        mean_difference = float(np.mean(absolute_difference))
        reproduced_mae = metrics(reproduced, truth)["mae"]
        archived_mae = metrics(full_prediction, truth)["mae"]
        verification = {
            "test_evaluated": False,
            "full_memory_prediction_maximum_absolute_difference": maximum_difference,
            "full_memory_prediction_mean_absolute_difference": mean_difference,
            "full_memory_validation_mae_reproduced": reproduced_mae,
            "full_memory_validation_mae_archived": archived_mae,
            "passed_exact_prediction_check": maximum_difference <= 1e-4,
        }
        with (output_dir / "verification.json").open("w", encoding="utf-8") as stream:
            json.dump(verification, stream, ensure_ascii=False, indent=2)
        log.info(
            "Full-memory reproduction: max difference %.8f, mean %.8f, MAE gap %+.8f",
            maximum_difference, mean_difference, reproduced_mae - archived_mae,
        )
        return
    prefix_end = int(0.55 * sample_count)
    gain = masked_node_gain(
        no_memory_prediction, full_prediction, truth, prefix_end
    )
    ranking = np.argsort(-gain, kind="stable")
    report = {
        "protocol": "same_checkpoint_sensor_utility_budget_pilot",
        "selection": "top sensor MAE gain on first 55% of Validation only",
        "evaluation": "last 45% of Validation, no Test metric",
        "full_cache": str(full_cache),
        "full_cache_bytes": full_cache.stat().st_size,
        "sample_count": sample_count,
        "selection_prefix_end": prefix_end,
        "test_evaluated": False,
        "no_memory_suffix": metrics(no_memory_prediction[prefix_end:], truth[prefix_end:]),
        "full_memory_suffix": metrics(full_prediction[prefix_end:], truth[prefix_end:]),
        "budgets": [],
    }
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        node_count = min(num_nodes, max(1, round(fraction * num_nodes)))
        node_ids = np.sort(ranking[:node_count]).astype(np.int64)
        if node_count == num_nodes:
            cache_path = full_cache
        else:
            cache_path = output_dir / f"sparse_memory_{node_count}nodes.npz"
            save_sparse_cache(full_cache, cache_path, node_ids)
        if node_count == num_nodes:
            # Score the already exported frozen full-memory checkpoint.
            prediction = full_prediction
        else:
            selected, arrays = expand_validation_cache(
                cache_path, sample_count, horizon, num_nodes
            )
            if not np.array_equal(selected, node_ids):
                raise ValueError("Sparse cache selected nodes differ")
            loader.retrieval_data, loader.exact_retrieval_data, loader.exact_available_data = arrays
            node_mask = torch.zeros(num_nodes, device=device)
            node_mask[torch.as_tensor(node_ids, device=device)] = 1.0
            mask_holder["value"] = node_mask.view(1, 1, num_nodes, 1)
            log.info("Forwarding budget %d/%d nodes on Validation ...", node_count, num_nodes)
            prediction, current_truth = collect(
                engine.model, loader, scaler, device, sample_count
            )
            if not np.array_equal(current_truth, truth):
                raise ValueError("Budget labels do not match")
        suffix_metrics = metrics(prediction[prefix_end:], truth[prefix_end:])
        item = {
            "selected_nodes": node_count,
            "fraction": node_count / num_nodes,
            "cache_path": str(cache_path),
            "cache_bytes": cache_path.stat().st_size,
            "cache_mib": cache_path.stat().st_size / 2**20,
            "suffix_metrics": suffix_metrics,
            "suffix_mae_over_full_memory": suffix_metrics["mae"] - report["full_memory_suffix"]["mae"],
        }
        report["budgets"].append(item)
        log.info(
            "Budget %d/%d: %.2f MiB, suffix MAE %.4f (full %+0.4f)",
            node_count, num_nodes, item["cache_mib"], suffix_metrics["mae"],
            item["suffix_mae_over_full_memory"],
        )
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    log.info("Saved %s", report_path)


if __name__ == "__main__":
    main()
