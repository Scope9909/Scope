"""Audit frozen final Test Top-k and exact-availability flips, without labels.

The selected package and exact comparator are immutable. Test observations are
used only as causal query histories. No checkpoint, prediction, or Test target
is opened, and no choice is made from the resulting rates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from adaptive_lossless_state_codec import load_runtime_state
from analog_distance import masked_history_distance
from utils.datasets_utils import file_path_dict, load_raw_data
from utils.online_residual_memory import OnlineResidualMemory

PACKAGES = HERE / "packages/online/adaptive_lossless_test_v1"
OUTPUT = HERE / "packages/online/frozen_test_retrieval_stability.json"
DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")


def restricted(state, node):
    result = dict(state)
    result["train_raw"] = np.ascontiguousarray(state["train_raw"][:, node:node + 1])
    result["node_ids"] = np.asarray(state["node_ids"])[node:node + 1]
    result["quant_scale"] = np.asarray(state["quant_scale"])[:, node:node + 1]
    for key in ("scaler_mean", "scaler_std"):
        value = np.asarray(state[key])
        result[key] = value if value.size == 1 else value[node:node + 1]
    return result


def audit_dataset(dataset):
    name = dataset.lower()
    exact_path = PACKAGES / f"{name}_test_exact.alsc"
    chosen_path = PACKAGES / f"{name}_test_selected.alsc"
    exact_state = load_runtime_state(exact_path)
    chosen_state = load_runtime_state(chosen_path)
    if set(exact_state) != set(chosen_state):
        raise AssertionError("State metadata keys changed")
    for key in exact_state:
        if key not in ("train_raw", "quant_scale") and not np.array_equal(exact_state[key], chosen_state[key]):
            raise AssertionError(f"Non-history state field {key} differs")
    different = np.flatnonzero(np.any(
        exact_state["train_raw"] != chosen_state["train_raw"], axis=0))
    # quant_scale may be serialized as float16 in the selected package; it
    # affects retrieved value scaling, but not top-k or exact-hit decisions.
    node_count = len(exact_state["node_ids"])
    with np.load(ROOT / "datasets" / dataset /
                 f"{dataset}_12to12_index_scaler.npz", allow_pickle=False) as archive:
        endpoints = np.asarray(archive["idx_test"], dtype=np.int64)
    if not len(different):
        return dict(dataset=dataset, test_queries=len(endpoints), nodes=node_count,
                    changed_local_node_indices=[], topk_set_flips=0,
                    topk_set_denominator=len(endpoints) * node_count,
                    topk_set_flip_rate=0.0, exact_availability_flips=0,
                    exact_availability_denominator=len(endpoints) * node_count * 12,
                    exact_availability_flip_rate=0.0,
                    reason="Decoded exact and selected runtime states are identical")
    raw, _ = load_raw_data(file_path_dict[dataset])
    with np.load(ROOT / "datasets" / dataset / f"{dataset}_time.npz",
                 allow_pickle=False) as archive:
        time = np.asarray(archive["time"])
    if endpoints.max() >= len(raw) or endpoints.max() >= len(time):
        raise ValueError("Test query endpoint outside raw observations")
    input_length = int(np.asarray(exact_state["input_length"]).item())
    horizon = int(np.asarray(exact_state["predict_length"]).item())
    topk_flips = exact_flips = 0
    for local_node in different:
        reference = OnlineResidualMemory(restricted(exact_state, int(local_node)))
        selected = OnlineResidualMemory(restricted(chosen_state, int(local_node)))
        if not (np.array_equal(reference.idx_train, selected.idx_train) and
                np.array_equal(reference.train_tod, selected.train_tod)):
            raise AssertionError("Retrieval pools changed")
        tod_all = np.asarray(time[endpoints, 0], dtype=np.int16)
        reference.validate_causality(endpoints, tod_all)
        node_id = int(reference.node_ids[0])
        offsets = np.arange(1 - input_length, 1, dtype=np.int64)
        for begin in range(0, len(endpoints), 64):
            subset = endpoints[begin:begin + 64]
            tod = tod_all[begin:begin + 64]
            queries = np.asarray(raw[subset[:, None] + offsets[None, :],
                                     node_id, 0], dtype=np.float32)[:, :, None]
            base_available = reference.query(queries, tod)[2]
            chosen_available = selected.query(queries, tod)[2]
            exact_flips += int(np.count_nonzero(base_available != chosen_available))
            for clock in np.unique(tod):
                query_positions = np.flatnonzero(tod == clock)
                candidate_positions = reference._train_positions_by_tod[int(clock)]
                candidate_endpoints = reference.idx_train[candidate_positions]
                base_history = reference.train_raw[
                    candidate_endpoints[:, None] + reference.x_offsets[None, :]]
                chosen_history = selected.train_raw[
                    candidate_endpoints[:, None] + selected.x_offsets[None, :]]
                base_distance = masked_history_distance(
                    queries[query_positions], base_history,
                    reference.min_valid_history, metric="masked_mse")
                chosen_distance = masked_history_distance(
                    queries[query_positions], chosen_history,
                    selected.min_valid_history, metric="masked_mse")
                k = reference.k
                base_top = np.sort(np.argpartition(
                    base_distance, kth=k - 1, axis=1)[:, :k, 0], axis=1)
                chosen_top = np.sort(np.argpartition(
                    chosen_distance, kth=k - 1, axis=1)[:, :k, 0], axis=1)
                topk_flips += int(np.count_nonzero(np.any(
                    base_top != chosen_top, axis=1)))
    return dict(dataset=dataset, test_queries=len(endpoints), nodes=node_count,
                changed_local_node_indices=different.tolist(),
                changed_global_node_ids=np.asarray(exact_state["node_ids"])[different].tolist(),
                topk_set_flips=topk_flips,
                topk_set_denominator=len(endpoints) * node_count,
                topk_set_flip_rate=topk_flips / (len(endpoints) * node_count),
                exact_availability_flips=exact_flips,
                exact_availability_denominator=len(endpoints) * node_count * horizon,
                exact_availability_flip_rate=exact_flips /
                    (len(endpoints) * node_count * horizon))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = []
    for dataset in DATASETS:
        row = audit_dataset(dataset)
        rows.append(row)
        print(dataset, row["topk_set_flip_rate"],
              row["exact_availability_flip_rate"], flush=True)
    OUTPUT.write_text(json.dumps(dict(
        scope="Frozen final Test retrieval stability audit",
        test_labels_used=False, test_predictions_used=False,
        selection_or_tuning_from_test=False,
        definitions=dict(topk="Unordered top-k candidate set per query and node",
                         exact="Exact-history availability per query, horizon and node"),
        rows=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
