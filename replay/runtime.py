"""Minimal shared runtime for the four INT8 frozen replay configurations."""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

from replay.model_runtime import build_model, unpack_dataloader_input  # noqa: E402

SELECTORS = {
    "PEMS03": "first55_three_block_stability_int8_topk36_budget",
    "PEMS04": "actual_online_state_byte_capped_int8_s1",
    "PEMS07": "first55_three_block_stability_int8_s1",
    "PEMS08": "actual_online_state_byte_capped_int8_s1",
}


def build_runtime(dataset: str):
    """Load one frozen checkpoint without consulting historical experiments."""
    from utils import datasets_utils

    final_runs = json.loads(
        (ROOT / "configs/final_runs.json").read_text(encoding="utf-8")
    )
    experiment = ROOT / final_runs[dataset]
    calibration = ROOT / "configs/frozen_calibration.json"
    for path in (experiment / "best.pth", experiment / "config.pkl", calibration):
        if not path.is_file():
            raise FileNotFoundError(path)

    # The pickle is a frozen local training configuration, not user input.
    with (experiment / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.pearson_adj = None
    if not (args.use_analog_lowrank_retrieval_residual and
            args.use_analog_exact_match_residual):
        raise ValueError("Frozen checkpoint lacks the expected retrieval adapters")

    # Dataset loading must not try to load historical query caches.
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger(f"scope_replay_{dataset.lower()}")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)

    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    model = build_model(args, device)
    model.load_state_dict(
        torch.load(experiment / "best.pth", map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False
    index_path = (
        ROOT / "datasets" / dataset /
        f"{dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz"
    )
    return loaders, scaler, model, device, index_path, calibration, SELECTORS[dataset]


def sparse_correction(model, prediction, ids, retrieval, exact, available):
    low = model.analog_lowrank_retrieval_residual
    exact_adapter = model.analog_exact_match_residual
    selected = prediction.index_select(2, ids)
    logits = low.horizon_logits[:, None] + torch.einsum(
        "nr,hr->hn", low.node_factor.index_select(0, ids), low.horizon_factor
    )
    blend = low.max_blend * torch.tanh(logits).view(1, logits.shape[0], len(ids), 1)
    selected = selected + blend * (retrieval - selected)
    exact_blend = (
        exact_adapter.max_blend * torch.tanh(exact_adapter.horizon_logits)
    ).view(1, -1, 1, 1)
    selected = selected + exact_blend * available.to(selected.dtype) * (exact - selected)
    return prediction.index_copy(2, ids, selected)


def collect(loader, endpoints, model, scaler, memory, device, batch_size):
    old_drop, old_batches, old_bs = loader.drop_last, loader.num_batch, loader.bs
    loader.bs = batch_size
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    predictions, truths = [], []
    cursor = 0
    ids = torch.as_tensor(memory.node_ids, dtype=torch.long, device=device)
    try:
        with torch.inference_mode():
            for loader_input, loader_target in loader.get_iterator():
                if cursor >= len(endpoints):
                    break
                x, x_time, *extras = unpack_dataloader_input(loader_input)
                if any(value is not None for value in extras):
                    raise ValueError("Precomputed query cache unexpectedly loaded")
                y, y_time = loader_target
                take = min(len(x), len(endpoints) - cursor)
                x, x_time = x[:take], x_time[:take]
                y, y_time = y[:take], y_time[:take]
                raw_history = np.rint(
                    scaler.inverse_transform(x)[..., 0][:, :, memory.node_ids]
                ).astype(np.float32)
                tod = np.asarray(x_time[:, -1, 0], dtype=np.int16)
                if memory.cold_start_endpoint is None:
                    retrieval, exact, available, _ = memory.query(raw_history, tod)
                else:
                    retrieval, exact, available, _ = memory.query_stream(
                        raw_history, endpoints[cursor:cursor + take], tod
                    )
                model_input = [
                    torch.as_tensor(x, dtype=torch.float32, device=device),
                    torch.as_tensor(x_time, dtype=torch.long, device=device),
                    torch.as_tensor(y_time, dtype=torch.long, device=device),
                ]
                prediction, _ = model(model_input)
                prediction = sparse_correction(
                    model,
                    prediction,
                    ids,
                    torch.as_tensor(retrieval, device=device),
                    torch.as_tensor(exact, device=device).float(),
                    torch.as_tensor(available, device=device),
                )
                predictions.append(
                    scaler.inverse_transform(prediction).cpu().numpy()[..., 0]
                )
                truths.append(np.asarray(y)[..., 0])
                cursor += take
    finally:
        loader.drop_last, loader.num_batch, loader.bs = old_drop, old_batches, old_bs
    if cursor != len(endpoints):
        raise ValueError("Loader did not cover the official split")
    return np.concatenate(predictions), np.concatenate(truths)
