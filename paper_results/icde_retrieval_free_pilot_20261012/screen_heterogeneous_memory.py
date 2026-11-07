"""Validation-only heterogeneous node memory pilot: off/INT8/FP16/FP32.

Uses one frozen checkpoint's native predictions and TRAIN-only compressed
per-node payload costs. This is a screening proxy, not physical deployment.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np


MODES = ("off", "int8", "fp16", "fp32")
FOLDS = ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0))
PAYLOAD = ("retrieval_train", "exact_retrieval_train", "exact_available_train")


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(np.asarray(archive[key]) for key in
                     ("val_prediction", "val_truth", "val_endpoints"))


def cost_by_node(path, nodes):
    result = np.zeros(nodes, dtype=np.int64)
    with np.load(path, allow_pickle=False) as archive:
        for key in PAYLOAD:
            array = archive[key]
            if array.ndim != 4 or array.shape[2] != nodes:
                raise ValueError(f"Unexpected {key} shape")
            for node in range(nodes):
                result[node] += len(zlib.compress(
                    np.ascontiguousarray(array[:, :, node, :]).tobytes(), 6))
            del array
        if "retrieval_quantization_scale" in archive.files:
            scale = archive["retrieval_quantization_scale"]
            for node in range(nodes):
                result[node] += len(zlib.compress(
                    np.ascontiguousarray(scale[:, node, :]).tobytes(), 6))
    return result


def loss_table(predictions, truth, start, end):
    target = truth[start:end]
    valid = np.round(target) > 0
    horizon_count = valid.sum(axis=(0, 2))
    if np.any(horizon_count == 0):
        raise ValueError("A horizon has no valid targets")
    losses = np.stack([
        (np.abs(prediction[start:end] - target) * valid).sum(axis=0)
        for prediction in predictions
    ], axis=-1)  # H,N,M
    return losses / horizon_count[:, None, None]


def utility(losses):
    return (losses[:, :, 0, None] - losses).mean(axis=0)


def allocate(values, costs, budget):
    modes = np.zeros(len(costs), dtype=np.int8)
    spent = 0
    while True:
        best = None
        for node in range(len(modes)):
            old = int(modes[node])
            for mode in range(1, len(MODES)):
                add_bytes = int(costs[node, mode] - costs[node, old])
                add_gain = float(values[node, mode] - values[node, old])
                if (add_gain < 0 or (add_gain == 0 and add_bytes >= 0) or
                        spent + add_bytes > budget):
                    continue
                score = float("inf") if add_bytes <= 0 else add_gain / add_bytes
                candidate = (score, add_gain, -add_bytes, -node, -mode)
                if best is None or candidate > best[0]:
                    best = (candidate, node, mode, add_bytes)
        if best is None:
            break
        _, node, mode, add_bytes = best
        modes[node] = mode
        spent += add_bytes
    return modes, spent


def score(losses, choices):
    chosen = np.take_along_axis(losses, choices[None, :, None], axis=2)[:, :, 0]
    return float(chosen.sum(axis=1).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-predictions", type=Path, required=True)
    p.add_argument("--fp32-predictions", type=Path, required=True)
    p.add_argument("--fp16-predictions", type=Path, required=True)
    p.add_argument("--int8-predictions", type=Path, required=True)
    p.add_argument("--fp32-cache", type=Path, required=True)
    p.add_argument("--fp16-cache", type=Path, required=True)
    p.add_argument("--int8-cache", type=Path, required=True)
    p.add_argument("--topk-nodes", type=int, default=184)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mixed-predictions-output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mixed_predictions_output is not None and args.mixed_predictions_output.exists():
        raise FileExistsError(args.mixed_predictions_output)
    paths = (args.backbone_predictions, args.int8_predictions,
             args.fp16_predictions, args.fp32_predictions)
    loaded = [load(path) for path in paths]
    truth, endpoints = loaded[0][1:]
    for path, (pred, target, end) in zip(paths, loaded):
        if pred.shape != truth.shape or not np.array_equal(target, truth) or not np.array_equal(end, endpoints):
            raise ValueError(f"Unaligned prediction archive: {path}")
    predictions = [item[0].astype(np.float64) for item in loaded]
    truth = truth.astype(np.float64)
    n, _, nodes = truth.shape
    if not 0 < args.topk_nodes <= nodes:
        raise ValueError("Invalid Top-k node count")
    costs = np.zeros((nodes, len(MODES)), dtype=np.int64)
    for mode, path in ((1, args.int8_cache), (2, args.fp16_cache), (3, args.fp32_cache)):
        costs[:, mode] = cost_by_node(path, nodes)
    folds = []
    for index, (fit_fraction, eval_fraction) in enumerate(FOLDS):
        fit_end, eval_end = int(n * fit_fraction), int(n * eval_fraction)
        fit_loss = loss_table(predictions, truth, 0, fit_end)
        fit_utility = utility(fit_loss)
        topk = np.argsort(-fit_utility[:, 1], kind="stable")[:args.topk_nodes]
        baseline = np.zeros(nodes, dtype=np.int8)
        baseline[topk] = 1
        budget = int(costs[topk, 1].sum())
        chosen, spent = allocate(fit_utility, costs, budget)
        if index == 0 and args.mixed_predictions_output is not None:
            mixed = np.empty_like(predictions[0], dtype=np.float32)
            for node, mode in enumerate(chosen):
                mixed[:, :, node] = predictions[int(mode)][:, :, node]
            args.mixed_predictions_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.mixed_predictions_output,
                                val_prediction=mixed, val_truth=truth.astype(np.float32),
                                val_endpoints=endpoints)
        eval_loss = loss_table(predictions, truth, fit_end, eval_end)
        baseline_mae, chosen_mae = score(eval_loss, baseline), score(eval_loss, chosen)
        folds.append({
            "fit_end": fit_end, "eval_start": fit_end, "eval_end": eval_end,
            "train_payload_budget_bytes": budget,
            "heterogeneous_train_payload_bytes": spent,
            "topk_int8_mae": baseline_mae,
            "heterogeneous_mae": chosen_mae,
            "heterogeneous_minus_topk_int8_mae": chosen_mae - baseline_mae,
            "mode_counts": {name: int(np.sum(chosen == mode))
                            for mode, name in enumerate(MODES)},
            "topk_int8_node_ids": sorted(int(x) for x in topk) if index == 0 else None,
            "heterogeneous_mode_by_node": chosen.tolist() if index == 0 else None,
        })
    report = {
        "protocol": "seed1_chronological_validation_heterogeneous_memory_screen",
        "test_evaluated": False,
        "cost_definition": "independent zlib-6 TRAIN query payload cost per node/mode; archive overhead excluded",
        "modes": MODES, "topk_nodes": args.topk_nodes,
        "folds": folds,
        "mean_heterogeneous_minus_topk_int8_mae": float(np.mean(
            [fold["heterogeneous_minus_topk_int8_mae"] for fold in folds])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"folds": [{key: value for key, value in fold.items()
                                  if key not in ("topk_int8_node_ids", "heterogeneous_mode_by_node")}
                                 for fold in folds],
                      "mean_heterogeneous_minus_topk_int8_mae":
                      report["mean_heterogeneous_minus_topk_int8_mae"]}, indent=2))


if __name__ == "__main__":
    main()
