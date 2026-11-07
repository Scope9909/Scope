"""Validation-only node-gain diagnostic for the frozen PEMS07 Student.

Uses the already frozen full-node exact .alsc state and SRM mask. The first
55% of Validation is not evaluated here; the last 45% is held out from node
selection and split into three chronological reporting folds. No Test data is
loaded, and this script does not change the frozen node set or paper results.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODEC_DIR = ROOT / "contrib3_index_codec_20261020"
sys.path.insert(0, str(CODEC_DIR))
import adaptive_lossless_state_codec as codec  # noqa: E402
import eval_hybrid_codec_test_once as previous  # noqa: E402

OUT = Path(__file__).resolve().parent / "pems07_exact_node_gain_validation.json"
DATASET = "PEMS07"
FULL_STATE = ROOT / "pems07_unified_ablation_20261028" / "pems07_full_exact_val.alsc"
SPARSE_STATE = CODEC_DIR / "packages" / "online" / "pems07_s1_val_u16.npz"


def node_gain(base: np.ndarray, full: np.ndarray, truth: np.ndarray,
              start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    target = truth[start:stop].astype(np.float64, copy=False)
    valid = np.rint(target) > 0
    count = valid.sum(axis=(0, 1), dtype=np.int64)
    base_error = (np.abs(base[start:stop] - target) * valid).sum(axis=(0, 1), dtype=np.float64)
    full_error = (np.abs(full[start:stop] - target) * valid).sum(axis=(0, 1), dtype=np.float64)
    gain = (base_error - full_error) / np.maximum(count, 1)
    return gain, count


def summarize(values: np.ndarray) -> dict:
    q = np.quantile(values, [0, 0.05, 0.25, 0.50, 0.75, 0.95, 1])
    return {
        "count": int(len(values)),
        "positive_count": int(np.sum(values > 0)),
        "negative_count": int(np.sum(values < 0)),
        "quantiles_0_5_25_50_75_95_100": [float(v) for v in q],
        "mean": float(np.mean(values)),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Will not overwrite prior diagnostic: {OUT}")
    if not FULL_STATE.is_file() or not SPARSE_STATE.is_file():
        raise FileNotFoundError("Frozen exact-state asset missing")

    runner = previous._load_runner(DATASET)
    exp_dir, _, _, _, _ = runner.inputs()
    with (exp_dir / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(v) for v in args.k_hops.split(",")]
    args.pearson_adj = None
    if not (args.use_analog_lowrank_retrieval_residual and
            args.use_analog_exact_match_residual):
        raise ValueError("Unexpected Student architecture")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("pems07_exact_node_gain_validation")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = runner.datasets_utils.load_dataset(args, logger)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(exp_dir / "best.pth", map_location=device,
                                            weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False
    with np.load(ROOT / "datasets" / DATASET / f"{DATASET}_12to12_index_scaler.npz",
                 allow_pickle=False) as archive:
        val_idx = np.asarray(archive["idx_val"], dtype=np.int64)
    with np.load(SPARSE_STATE, allow_pickle=False) as archive:
        selected_ids = np.asarray(archive["node_ids"], dtype=np.int64)
    selected = np.zeros(nodes, dtype=bool)
    selected[selected_ids] = True
    memory = runner.OnlineResidualMemory(codec.load_runtime_state(FULL_STATE))
    if memory.cold_start_endpoint != int(val_idx[0]) or len(memory.node_ids) != nodes:
        raise ValueError("Full exact state does not match official Validation")
    print("Collecting full-node exact Validation predictions", flush=True)
    full, truth = runner.collect(loaders["val_loader"], val_idx, model, scaler,
                                 memory, device, 32)
    print("Collecting base Validation predictions", flush=True)
    base, base_truth = runner.collect(loaders["val_loader"], val_idx, model, scaler,
                                      None, device, 32)
    if full.shape != base.shape or not np.array_equal(truth, base_truth):
        raise AssertionError("Validation predictions are not aligned")
    n = len(val_idx)
    edges = [int(n * f) for f in (0.55, 0.70, 0.85, 1.0)]
    folds = []
    for number, (start, stop) in enumerate(zip(edges[:-1], edges[1:]), 1):
        gain, count = node_gain(base, full, truth, start, stop)
        folds.append({
            "fold": number, "start": start, "stop": stop,
            "selected": summarize(gain[selected]),
            "unselected": summarize(gain[~selected]),
            "selected_gain": [float(v) for v in gain[selected]],
            "unselected_gain": [float(v) for v in gain[~selected]],
            "valid_count_by_node": [int(v) for v in count],
        })
    gain, count = node_gain(base, full, truth, edges[0], edges[-1])
    result = {
        "protocol": "frozen PEMS07 Student, exact full-node state, validation-only held-out node gain",
        "test_evaluated": False,
        "checkpoint": str(exp_dir / "best.pth"),
        "full_state": str(FULL_STATE),
        "srm_node_source": str(SPARSE_STATE),
        "student_parameters": 703330,
        "selected_nodes": int(selected.sum()),
        "all_nodes": int(nodes),
        "selection_prefix_samples": int(edges[0]),
        "folds": folds,
        "heldout_selected": summarize(gain[selected]),
        "heldout_unselected": summarize(gain[~selected]),
        "heldout_gain_by_node": [float(v) for v in gain],
        "heldout_valid_count_by_node": [int(v) for v in count],
        "selected_node_ids": [int(v) for v in selected_ids],
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("protocol", "heldout_selected", "heldout_unselected")}, indent=2),
          flush=True)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
