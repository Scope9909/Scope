"""Exploratory Validation-only node allocator under an actual online-state cap.

Ranking uses early-Validation raw MAE gain per compressed Train-series proxy
byte. Feasibility uses the exact serialized Train-only state plus frozen
checkpoint and calibration bytes. It is a greedy baseline, not an optimal
solver or an independent novelty claim.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from audit_online_train_memory import packed_state  # noqa: E402
from screen_heterogeneous_memory import FOLDS, load, loss_table, score, utility  # noqa: E402
from utils.datasets_utils import file_path_dict, load_raw_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("PEMS04", "PEMS08"), required=True)
    parser.add_argument("--backbone-predictions", type=Path, required=True)
    parser.add_argument("--int8-predictions", type=Path, required=True)
    parser.add_argument("--full-int8-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--total-byte-budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    checkpoint_bytes = args.checkpoint.stat().st_size
    calibration_bytes = args.calibration.stat().st_size
    fixed_bytes = checkpoint_bytes + calibration_bytes
    if args.total_byte_budget <= fixed_bytes:
        raise ValueError("Total budget cannot even fit the checkpoint and calibration")

    backbone, truth, endpoints = load(args.backbone_predictions)
    full, truth2, endpoints2 = load(args.int8_predictions)
    if not (backbone.shape == full.shape == truth.shape and
            np.array_equal(truth, truth2) and np.array_equal(endpoints, endpoints2)):
        raise ValueError("Predictions, truth, or endpoints are not strictly aligned")
    count, _, nodes = truth.shape
    predictions = [backbone.astype(np.float64), full.astype(np.float64)]
    truth = truth.astype(np.float64)
    fit_end = int(count * FOLDS[0][0])
    gains = utility(loss_table(predictions, truth, 0, fit_end))[:, 1]

    with np.load(args.full_int8_cache, allow_pickle=False) as archive:
        if not np.array_equal(np.asarray(archive["idx_val"]), endpoints):
            raise ValueError("Cache Validation endpoints differ from predictions")
        if str(archive["retrieval_quantization_fit_split"].item()) != "train_only":
            raise ValueError("Quantization scale is not Train-fitted")
        if str(archive["exact_match_rule"].item()) != "masked_mse_eq_0":
            raise ValueError("Unexpected exact-history rule")
        if ("calendar_mode" in archive.files and
                str(archive["calendar_mode"].item()) != "tod"):
            raise ValueError("Only same-time-of-day retrieval is supported")
        if ("aggregation" in archive.files and
                str(archive["aggregation"].item()) != "mean"):
            raise ValueError("Only mean aggregation is supported")
        idx_train = np.asarray(archive["idx_train"], dtype=np.int64)
        horizon = int(archive["predict_length"])
        length = int(archive["input_length"])
        scale = np.asarray(archive["retrieval_quantization_scale"])
        mean = np.asarray(archive["scaler_mean"])
        std = np.asarray(archive["scaler_std"])
        metadata = {key: np.asarray(archive[key]) for key in
                    ("k", "min_valid_history", "input_length", "predict_length",
                     "exact_min_valid_history")}
    data, _ = load_raw_data(file_path_dict[args.dataset])
    with np.load(ROOT / "datasets" / args.dataset / f"{args.dataset}_time.npz",
                 allow_pickle=False) as archive:
        time_data = np.asarray(archive["time"])
    if len(data) != len(time_data) or data.shape[1] != nodes or scale.shape[1] != nodes:
        raise ValueError("Raw data/time/cache node dimensions differ")
    if idx_train.min() < length - 1 or idx_train.max() + horizon >= len(data):
        raise ValueError("Invalid Train endpoint span")
    last_train_frame = int(idx_train.max() + horizon)
    train_raw = np.asarray(data[:last_train_frame + 1, :, 0], dtype=np.float32)
    train_tod = np.asarray(time_data[idx_train, 0], dtype=np.int16)
    shared = {"idx_train": idx_train.astype(np.int32), "train_tod": train_tod,
              **metadata}

    def state_for(ids: np.ndarray) -> dict[str, np.ndarray]:
        ids = np.sort(np.asarray(ids, dtype=np.int32))
        return {
            "train_raw": np.ascontiguousarray(train_raw[:, ids]),
            "idx_train": shared["idx_train"],
            "train_tod": shared["train_tod"],
            "node_ids": ids,
            "quant_scale": np.ascontiguousarray(scale[:, ids, :]),
            "scaler_mean": mean if mean.size == 1 else mean[ids],
            "scaler_std": std if std.size == 1 else std[ids],
            **metadata,
        }

    empty_bytes = len(packed_state(state_for(np.empty(0, dtype=np.int32))))
    if args.total_byte_budget <= fixed_bytes + empty_bytes:
        raise ValueError("Total budget cannot fit a nonempty Train-only state")
    proxy = np.asarray([
        len(zlib.compress(np.ascontiguousarray(train_raw[:, node]).tobytes(), 6)) +
        len(zlib.compress(np.ascontiguousarray(scale[:, node, :]).tobytes(), 6))
        for node in range(nodes)], dtype=np.int64)
    if np.any(proxy <= 0):
        raise ValueError("Invalid per-node Train-state proxy")
    order = np.lexsort((np.arange(nodes), -gains / proxy))
    selected: list[int] = []
    actual_state_bytes = empty_bytes
    examined = 0
    for node in order:
        node = int(node)
        if gains[node] <= 0:
            break
        examined += 1
        candidate = np.asarray([*selected, node], dtype=np.int32)
        candidate_bytes = len(packed_state(state_for(candidate)))
        if fixed_bytes + candidate_bytes <= args.total_byte_budget:
            selected.append(node)
            actual_state_bytes = candidate_bytes
    if not selected:
        raise ValueError("No positive-utility node fits the online-state budget")
    state_bytes = packed_state(state_for(np.asarray(selected, dtype=np.int32)))
    if len(state_bytes) != actual_state_bytes:
        raise AssertionError("Serialized state size changed after selection")
    choices = np.zeros(nodes, dtype=np.int8)
    choices[selected] = 1
    folds = []
    for start_fraction, end_fraction in FOLDS:
        start, end = int(count * start_fraction), int(count * end_fraction)
        folds.append({"eval_start": start, "eval_end": end,
                      "selected_raw_mae": score(loss_table(predictions, truth,
                                                           start, end), choices)})
    total_bytes = fixed_bytes + len(state_bytes)
    plan = {
        "protocol": "actual_online_state_byte_capped_int8",
        "test_evaluated": False,
        "cost_definition": "physical Train-only state archive plus checkpoint and calibration files",
        "total_byte_budget": args.total_byte_budget,
        "selected_total_bytes": total_bytes,
        "online_state_bytes": len(state_bytes),
        "folds": [{"fit_end": fit_end,
                   "topk_int8_node_ids": sorted(selected),
                   "heterogeneous_mode_by_node": choices.tolist()}],
    }
    report = {
        "protocol": "online_state_byte_capped_validation_screen",
        "dataset": args.dataset,
        "test_evaluated": False,
        "selection_prefix_samples": fit_end,
        "total_byte_budget": args.total_byte_budget,
        "selected_total_bytes": total_bytes,
        "budget_satisfied": total_bytes <= args.total_byte_budget,
        "checkpoint_bytes": checkpoint_bytes,
        "calibration_bytes": calibration_bytes,
        "selected_online_state_bytes": len(state_bytes),
        "empty_online_state_bytes": empty_bytes,
        "selected_nodes": len(selected),
        "positive_nodes_examined": examined,
        "folds": folds,
        "mean_raw_mae": float(np.mean([item["selected_raw_mae"] for item in folds])),
        "note": "No Validation query result in state; exact-scan retrieval, not an end-to-end serving latency audit",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (args.output_dir / "screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "online_train_state.npz").write_bytes(state_bytes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
