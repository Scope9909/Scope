"""Export one selected checkpoint to the common ValCal archive schema.

The checkpoint must already have been selected using Validation.  This script
does no model selection and performs exactly one deterministic forward pass on
Validation and Test.  Predictions are saved in raw traffic-flow units together
with the official split endpoints and labels.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


CLI_ARGS = parse_args()
# run.py owns a legacy global parser; do not expose this script's arguments.
sys.argv = [sys.argv[0]]

from run import trainer, unpack_dataloader_input  # noqa: E402
from utils import datasets_utils  # noqa: E402


def logger() -> logging.Logger:
    result = logging.getLogger("export_checkpoint_predictions")
    result.setLevel(logging.INFO)
    if not result.handlers:
        result.addHandler(logging.StreamHandler(sys.stdout))
    return result


@torch.no_grad()
def collect(model, loader, scaler, device, sample_count):
    predictions = []
    truths = []
    original_drop_last = loader.drop_last
    original_num_batch = loader.num_batch
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    try:
        model.eval()
        for loader_input, loader_target in loader.get_iterator():
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
                if value is not None:
                    model_input.append(torch.as_tensor(value, device=device))
            prediction, _ = model(model_input)
            predictions.append(prediction.cpu())
            truths.append(torch.as_tensor(y, dtype=torch.float32))
    finally:
        loader.drop_last = original_drop_last
        loader.num_batch = original_num_batch

    prediction = torch.cat(predictions, dim=0)[:sample_count]
    truth = torch.cat(truths, dim=0)[:sample_count]
    prediction = scaler.inverse_transform(prediction).numpy()[..., 0]
    truth = truth.numpy()[..., 0]
    if prediction.shape != truth.shape:
        raise RuntimeError(f"prediction/truth mismatch: {prediction.shape} != {truth.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise FloatingPointError("export contains non-finite values")
    return prediction.astype(np.float32), truth.astype(np.float32)


def main() -> None:
    log = logger()
    exp_dir = CLI_ARGS.exp_dir.resolve()
    output = CLI_ARGS.output.resolve()
    config_path = exp_dir / "config.pkl"
    checkpoint_path = exp_dir / "best.pth"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"config.pkl/best.pth missing under {exp_dir}")

    with config_path.open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    needs_graph = bool(getattr(args, "use_graph_conv", 0)) or (
        getattr(args, "spatial_type", None) == "agcrn"
    )
    pearson_path = getattr(args, "pearson_adj_path", None)
    default_pearson_path = Path("datasets") / args.dataset_name / (
        f"{args.dataset_name}_pearson_adj.npy"
    )
    if needs_graph and pearson_path and Path(pearson_path).is_file():
        args.pearson_adj = torch.as_tensor(
            np.load(pearson_path), dtype=torch.float32
        )
    elif needs_graph and default_pearson_path.is_file():
        args.pearson_adj = torch.as_tensor(
            np.load(default_pearson_path), dtype=torch.float32
        )
    else:
        args.pearson_adj = None

    index_path = Path("datasets") / args.dataset_name / (
        f"{args.dataset_name}_{args.input_length}to{args.predict_length}_index_scaler.npz"
    )
    with np.load(index_path, allow_pickle=False) as archive:
        endpoints = {
            split: np.asarray(archive[f"idx_{split}"], dtype=np.int64)
            for split in ("val", "test")
        }
    if any(not np.all(np.diff(values) > 0) for values in endpoints.values()):
        raise ValueError("official split endpoints must be strictly increasing")

    device = torch.device(args.device)
    dataloader, num_nodes, scaler, _ = datasets_utils.load_dataset(args, log)
    args.num_nodes = num_nodes
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    state = torch.load(checkpoint_path, map_location=device)
    engine.model.load_state_dict(state, strict=True)

    values = {}
    for split in ("val", "test"):
        log.info("Forward %s once from selected checkpoint ...", split)
        prediction, truth = collect(
            engine.model,
            dataloader[f"{split}_loader"],
            scaler,
            device,
            len(endpoints[split]),
        )
        values[f"{split}_endpoints"] = endpoints[split]
        values[f"{split}_prediction"] = prediction
        values[f"{split}_truth"] = truth

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, output)
    log.info("Saved %s", output)


if __name__ == "__main__":
    main()
