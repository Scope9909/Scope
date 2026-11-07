"""Audit cross-horizon gradient conflict in a trained lightweight Student.

The audit is deliberately diagnostic only.  It loads a frozen deployment
checkpoint, enables gradients solely for the shared predictor trunk, and
measures cosine similarity between the twelve horizon-specific supervised
MAE gradients on Train batches.  Validation and Test labels are never read.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models import TLAST_improved
from utils import datasets_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gradient-scope",
        choices=("predictor_shared", "core_shared"),
        default="predictor_shared",
        help=(
            "predictor_shared audits the factorized predictor trunk only; "
            "core_shared additionally audits the lightweight backbone while "
            "excluding horizon heads and all Hist2Res memory modules"
        ),
    )
    return parser.parse_args()


def unpack_loader_input(loader_input):
    empty_candidate = (None, None, None)
    if len(loader_input) == 2:
        x, x_time = loader_input
        return (x, x_time, None, None, None) + empty_candidate
    if len(loader_input) == 3:
        x, x_time, analog = loader_input
        return (x, x_time, analog, None, None) + empty_candidate
    if len(loader_input) == 5:
        x, x_time, analog, exact, available = loader_input
        return (x, x_time, analog, exact, available) + empty_candidate
    if len(loader_input) == 8:
        return tuple(loader_input)
    raise ValueError(f"Unsupported loader input count: {len(loader_input)}")


def tensorize_model_input(loader_input, y_time, device):
    (
        x,
        x_time,
        analog,
        exact,
        exact_available,
        candidate_future,
        candidate_available,
        candidate_features,
    ) = unpack_loader_input(loader_input)
    model_input = [
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(x_time, dtype=torch.long, device=device),
        torch.as_tensor(y_time, dtype=torch.long, device=device),
    ]
    if analog is not None:
        model_input.append(torch.as_tensor(analog, device=device))
    if exact is not None:
        model_input.extend([
            torch.as_tensor(exact, device=device),
            torch.as_tensor(exact_available, device=device),
        ])
    if candidate_future is not None:
        model_input.extend([
            torch.as_tensor(candidate_future, dtype=torch.float32, device=device),
            torch.as_tensor(candidate_available, dtype=torch.float32, device=device),
            torch.as_tensor(candidate_features, dtype=torch.float32, device=device),
        ])
    return model_input


def masked_horizon_mae(prediction, target, horizon):
    pred_h = prediction[:, horizon]
    target_h = target[:, horizon]
    valid = torch.isfinite(target_h) & target_h.ne(0.0)
    if not bool(valid.any()):
        raise RuntimeError(f"Horizon {horizon + 1} has no valid Train labels")
    return torch.abs(pred_h[valid] - target_h[valid]).mean()


def gradient_dot(left, right):
    return sum((a * b).sum() for a, b in zip(left, right))


def gradient_norm(gradient):
    return torch.sqrt(
        sum(tensor.square().sum() for tensor in gradient).clamp_min(1e-30)
    )


def main() -> None:
    cli = parse_args()
    if cli.batches <= 0 or cli.batch_size <= 0:
        raise ValueError("--batches and --batch-size must be positive")
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    config_path = Path(cli.config_json).resolve()
    checkpoint_path = Path(cli.checkpoint).resolve()
    output_path = Path(cli.output).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        saved_config = json.load(handle)
    args = SimpleNamespace(**saved_config)
    args.device = cli.device
    args.batch_size = cli.batch_size
    args.cache_data_path = "datasets"
    # The diagnostic uses ordinary Train labels only.  Avoid loading the
    # teacher cache; retrieval inputs remain enabled because they are part of
    # the deployed Student function represented by this checkpoint.
    args.kd_alpha = 0.0
    args.teacher_predictions_path = "none"
    args.teacher_predictions_strict_alignment = 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("horizon-gradient-audit")
    dataloader, num_nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.num_nodes = num_nodes
    args.prob_factor = getattr(args, "prob_factor", 5)
    args.pearson_adj = None
    if isinstance(args.k_hops, str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]

    device = torch.device(cli.device)
    model = TLAST_improved.Model(args).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: {missing=}, {unexpected=}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    predictor = model.time_wise_predictor
    if not hasattr(predictor, "predict_unit"):
        raise TypeError("Active Student predictor has no shared predict_unit")
    shared_parameters = []
    shared_parameter_names = []
    if cli.gradient_scope == "predictor_shared":
        selected_parameters = (
            ("time_wise_predictor.predict_unit." + name, parameter)
            for name, parameter in predictor.predict_unit.named_parameters()
        )
    else:
        # The proposed contribution must remain independent of Hist2Res, so
        # every analog-memory module stays frozen.  Horizon-specific readouts
        # are also excluded because they do not share parameters and hence
        # cannot exhibit multi-task gradient conflict.
        selected_parameters = (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if not name.startswith("analog_")
            and not name.startswith(
                "time_wise_predictor.predict_unit_list."
            )
        )
    for name, parameter in selected_parameters:
        parameter.requires_grad_(True)
        shared_parameters.append(parameter)
        shared_parameter_names.append(name)
    if not shared_parameters:
        raise RuntimeError("No shared predictor parameters were selected")

    horizon_count = int(args.predict_length)
    cosine_sum = torch.zeros(
        horizon_count, horizon_count, dtype=torch.float64, device=device
    )
    negative_count = torch.zeros_like(cosine_sum)
    loss_sum = torch.zeros(horizon_count, dtype=torch.float64, device=device)
    norm_sum = torch.zeros_like(loss_sum)
    batches_used = 0

    dataloader["train_loader"].shuffle()
    for loader_input, loader_target in dataloader["train_loader"].get_iterator():
        y, y_time = loader_target
        model_input = tensorize_model_input(loader_input, y_time, device)
        target = torch.as_tensor(y, dtype=torch.float32, device=device)
        output, _ = model(model_input)
        prediction = scaler.inverse_transform(output)

        losses = [
            masked_horizon_mae(prediction, target, horizon)
            for horizon in range(horizon_count)
        ]
        gradients = []
        norms = []
        for horizon, loss in enumerate(losses):
            grad = torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=(horizon + 1 < horizon_count),
                allow_unused=False,
            )
            detached = tuple(value.detach() for value in grad)
            gradients.append(detached)
            norms.append(gradient_norm(detached))
            loss_sum[horizon] += loss.detach().double()
            norm_sum[horizon] += norms[-1].double()

        for left in range(horizon_count):
            for right in range(horizon_count):
                cosine = gradient_dot(
                    gradients[left], gradients[right]
                ) / (norms[left] * norms[right]).clamp_min(1e-30)
                cosine_sum[left, right] += cosine.double()
                negative_count[left, right] += (cosine < 0).double()

        batches_used += 1
        logger.info(
            "audited Train batch %d/%d; mean horizon MAE=%.6f",
            batches_used,
            cli.batches,
            float(torch.stack(losses).mean()),
        )
        del prediction, output, gradients, losses
        if batches_used >= cli.batches:
            break

    if batches_used == 0:
        raise RuntimeError("Train loader produced no batches")
    cosine_mean = (cosine_sum / batches_used).cpu().numpy()
    negative_rate = (negative_count / batches_used).cpu().numpy()
    mean_losses = (loss_sum / batches_used).cpu().numpy()
    mean_norms = (norm_sum / batches_used).cpu().numpy()
    off_diagonal = ~np.eye(horizon_count, dtype=bool)
    upper_triangle = np.triu(np.ones_like(off_diagonal), k=1).astype(bool)
    adjacent = np.eye(horizon_count, k=1, dtype=bool)
    group_labels = [
        "short" if index < 4 else "mid" if index < 8 else "long"
        for index in range(horizon_count)
    ]
    cross_group = np.fromfunction(
        np.vectorize(
            lambda i, j: group_labels[int(i)] != group_labels[int(j)],
            otypes=[bool],
        ),
        (horizon_count, horizon_count),
        dtype=int,
    ) & upper_triangle

    result = {
        "protocol": {
            "split": "Train only",
            "checkpoint": str(checkpoint_path),
            "config_json": str(config_path),
            "device": str(device),
            "seed": cli.seed,
            "batch_size": cli.batch_size,
            "batches_requested": cli.batches,
            "batches_used": batches_used,
            "gradient_scope": cli.gradient_scope,
            "parameter_names": shared_parameter_names,
            "parameter_count": int(sum(p.numel() for p in shared_parameters)),
            "loss": "per-horizon masked MAE in raw traffic units",
        },
        "mean_horizon_mae": mean_losses.tolist(),
        "mean_gradient_norm": mean_norms.tolist(),
        "mean_cosine_matrix": cosine_mean.tolist(),
        "batch_negative_rate_matrix": negative_rate.tolist(),
        "summary": {
            "mean_off_diagonal_cosine": float(cosine_mean[off_diagonal].mean()),
            "minimum_pair_mean_cosine": float(cosine_mean[upper_triangle].min()),
            "pairs_with_negative_mean_cosine": int(
                np.sum(cosine_mean[upper_triangle] < 0)
            ),
            "pair_count": int(np.sum(upper_triangle)),
            "negative_mean_pair_fraction": float(
                np.mean(cosine_mean[upper_triangle] < 0)
            ),
            "mean_pair_batch_negative_rate": float(
                negative_rate[upper_triangle].mean()
            ),
            "adjacent_mean_cosine": float(cosine_mean[adjacent].mean()),
            "adjacent_batch_negative_rate": float(
                negative_rate[adjacent].mean()
            ),
            "cross_group_mean_cosine": float(cosine_mean[cross_group].mean()),
            "cross_group_batch_negative_rate": float(
                negative_rate[cross_group].mean()
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    logger.info("wrote %s", output_path)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
