"""Train-only online residual memory used by deployment-style validation.

The serialized state contains only Train observations/endpoints, calendar
indices, node ids, Train-fitted quantization scales, and dataset scaling
statistics. Query outputs and Validation/Test labels are never serialized.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Mapping

import numpy as np

from analog_distance import masked_history_distance
from utils.datasets_utils import file_path_dict, load_raw_data


REQUIRED_KEYS = {
    "train_raw", "idx_train", "train_tod", "node_ids", "quant_scale",
    "scaler_mean", "scaler_std", "k", "min_valid_history",
    "input_length", "predict_length", "exact_min_valid_history",
}


def build_cold_start_state(state: Mapping[str, np.ndarray],
                           first_query_endpoint: int) -> dict[str, np.ndarray]:
    """Create a causal snapshot plus pending Train endpoints for stream release.

    The snapshot is taken immediately after the observation at
    ``first_query_endpoint`` arrives. Candidate futures beyond that point are not
    serialized. Their endpoint/TOD metadata is harmless and lets the runtime
    activate them once the missing observations arrive in the live query stream.
    """
    missing = REQUIRED_KEYS.difference(state)
    if missing:
        raise ValueError(f"Source state is missing keys: {sorted(missing)}")
    result = {key: np.asarray(value) for key, value in state.items()
              if key not in {"pending_idx_train", "pending_train_tod",
                             "cold_start_endpoint", "source_candidate_count"}}
    idx_train = np.asarray(result["idx_train"], dtype=np.int64)
    train_tod = np.asarray(result["train_tod"], dtype=np.int16)
    horizon = int(np.asarray(result["predict_length"]).item())
    endpoint = int(first_query_endpoint)
    if endpoint < 0 or endpoint >= len(result["train_raw"]):
        raise ValueError("Cold-start endpoint is outside the source state")
    active = idx_train + horizon <= endpoint
    if not np.any(active):
        raise ValueError("Cold-start embargo removes every Train candidate")
    result["idx_train"] = idx_train[active].astype(np.int32)
    result["train_tod"] = train_tod[active]
    result["pending_idx_train"] = idx_train[~active].astype(np.int32)
    result["pending_train_tod"] = train_tod[~active]
    result["train_raw"] = np.ascontiguousarray(
        np.asarray(result["train_raw"])[:endpoint + 1])
    result["cold_start_endpoint"] = np.asarray(endpoint, dtype=np.int64)
    result["source_candidate_count"] = np.asarray(len(idx_train), dtype=np.int64)
    return result


def packed_state(state: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **state)
    return buffer.getvalue()


def _nonzero_mean(values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    valid = values != 0
    total = np.where(valid, values, 0.0).sum(axis=axis, dtype=np.float64)
    count = valid.sum(axis=axis)
    mean = np.zeros_like(total, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return mean, count


def build_train_only_state(dataset: str, full_int8_cache: Path,
                           node_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Build the exact deployable state without reading query targets."""
    node_ids = np.asarray(node_ids, dtype=np.int32)
    with np.load(full_int8_cache, allow_pickle=False) as archive:
        retrieval_format = "int8" if "retrieval_quantization_scale" in archive.files else "float32"
        if retrieval_format == "int8" and str(np.asarray(
                archive["retrieval_quantization_fit_split"]).item()) != "train_only":
            raise ValueError("Retrieval quantization must be fitted on Train only")
        if str(np.asarray(archive["exact_match_rule"]).item()) != "masked_mse_eq_0":
            raise ValueError("Unsupported exact-history rule")
        if "calendar_mode" in archive.files and str(archive["calendar_mode"].item()) != "tod":
            raise ValueError("Only same-time-of-day retrieval is supported")
        if "aggregation" in archive.files and str(archive["aggregation"].item()) != "mean":
            raise ValueError("Only mean aggregation is supported")
        idx_train = np.asarray(archive["idx_train"], dtype=np.int64)
        scale_all = (np.asarray(archive["retrieval_quantization_scale"], dtype=np.float32)
                     if retrieval_format == "int8" else None)
        scaler_mean = np.asarray(archive["scaler_mean"])
        scaler_std = np.asarray(archive["scaler_std"])
        metadata = {key: np.asarray(archive[key]) for key in (
            "k", "min_valid_history", "input_length", "predict_length",
            "exact_min_valid_history")}
    data, _ = load_raw_data(file_path_dict[dataset])
    time_path = Path("datasets") / dataset / f"{dataset}_time.npz"
    with np.load(time_path, allow_pickle=False) as archive:
        time_data = np.asarray(archive["time"])
    if (node_ids.ndim != 1 or len(node_ids) == 0 or
            len(np.unique(node_ids)) != len(node_ids) or
            np.any((node_ids < 0) | (node_ids >= data.shape[1]))):
        raise ValueError("node_ids must be unique valid dataset node indices")
    horizon = int(metadata["predict_length"])
    last_train_frame = int(idx_train.max() + horizon)
    if last_train_frame >= len(data):
        raise ValueError("Train endpoint future exceeds raw data")
    node_mean = scaler_mean if scaler_mean.size == 1 else scaler_mean[node_ids]
    node_std = scaler_std if scaler_std.size == 1 else scaler_std[node_ids]
    return {
        "train_raw": np.ascontiguousarray(
            data[:last_train_frame + 1, node_ids, 0], dtype=np.float32),
        "idx_train": idx_train.astype(np.int32),
        "train_tod": np.asarray(time_data[idx_train, 0], dtype=np.int16),
        "node_ids": node_ids,
        "quant_scale": (np.ascontiguousarray(scale_all[:, node_ids, :])
                        if scale_all is not None else
                        np.ones((horizon, len(node_ids), 1), dtype=np.float32)),
        "retrieval_format": np.asarray(retrieval_format),
        "scaler_mean": np.asarray(node_mean),
        "scaler_std": np.asarray(node_std),
        **metadata,
    }


class OnlineResidualMemory:
    """Exact-scan online retrieval backed only by a Train state archive."""

    def __init__(self, state: Mapping[str, np.ndarray]):
        missing = REQUIRED_KEYS.difference(state)
        if missing:
            raise ValueError(f"Online state is missing keys: {sorted(missing)}")
        self.train_raw = np.asarray(state["train_raw"], dtype=np.float32)
        self.idx_train = np.asarray(state["idx_train"], dtype=np.int64)
        self.train_tod = np.asarray(state["train_tod"], dtype=np.int16)
        self.node_ids = np.asarray(state["node_ids"], dtype=np.int64)
        self.quant_scale = np.asarray(state["quant_scale"], dtype=np.float32)
        self.retrieval_format = str(np.asarray(
            state.get("retrieval_format", "int8")).item())
        self.scaler_mean = np.asarray(state["scaler_mean"])
        self.scaler_std = np.asarray(state["scaler_std"])
        self.k = int(np.asarray(state["k"]).item())
        self.min_valid_history = int(np.asarray(state["min_valid_history"]).item())
        self.input_length = int(np.asarray(state["input_length"]).item())
        self.predict_length = int(np.asarray(state["predict_length"]).item())
        self.exact_min_valid_history = int(
            np.asarray(state["exact_min_valid_history"]).item())
        self.pending_idx_train = np.asarray(
            state.get("pending_idx_train", np.empty(0, dtype=np.int32)),
            dtype=np.int64)
        self.pending_train_tod = np.asarray(
            state.get("pending_train_tod", np.empty(0, dtype=np.int16)),
            dtype=np.int16)
        self.cold_start_endpoint = (None if "cold_start_endpoint" not in state else
                                    int(np.asarray(state["cold_start_endpoint"]).item()))
        self.source_candidate_count = int(np.asarray(
            state.get("source_candidate_count", len(self.idx_train))).item())
        self.x_offsets = np.arange(1 - self.input_length, 1, dtype=np.int64)
        self.y_offsets = np.arange(1, self.predict_length + 1, dtype=np.int64)
        self._validate()
        self._rebuild_indices()

    @classmethod
    def from_npz(cls, path: Path | str) -> "OnlineResidualMemory":
        with np.load(path, allow_pickle=False) as archive:
            return cls({key: archive[key] for key in archive.files})

    def _validate(self) -> None:
        nodes = len(self.node_ids)
        if (self.train_raw.ndim != 2 or self.train_raw.shape[1] != nodes or
                self.idx_train.ndim != 1 or self.train_tod.shape != self.idx_train.shape):
            raise ValueError("Malformed Train-only state dimensions")
        if (len(np.unique(self.node_ids)) != nodes or np.any(self.node_ids < 0) or
                np.any(self.idx_train < self.input_length - 1) or
                np.any(self.idx_train + self.predict_length >= len(self.train_raw))):
            raise ValueError("Invalid node ids or Train endpoints")
        expected_scale = (self.predict_length, nodes, 1)
        if (self.quant_scale.shape != expected_scale or
                not np.isfinite(self.quant_scale).all() or
                np.any(self.quant_scale <= 0)):
            raise ValueError("Invalid Train-fitted INT8 scale")
        if self.scaler_std.size not in (1, nodes) or np.any(self.scaler_std <= 0):
            raise ValueError("Scaler must be global or node-wise and positive")
        if self.scaler_mean.size not in (1, nodes):
            raise ValueError("Scaler mean must be global or node-wise")
        if self.k < 1:
            raise ValueError("k must be positive")
        if self.retrieval_format not in ("int8", "float32"):
            raise ValueError("Unsupported retrieval format")
        if self.pending_idx_train.shape != self.pending_train_tod.shape:
            raise ValueError("Pending endpoint/time arrays differ")
        if len(self.pending_idx_train):
            if (self.cold_start_endpoint is None or
                    np.any(self.pending_idx_train < self.input_length - 1) or
                    np.any(self.pending_idx_train >= len(self.train_raw)) or
                    np.any(self.pending_idx_train + self.predict_length <=
                           self.cold_start_endpoint)):
                raise ValueError("Malformed cold-start pending candidates")
            if np.any(self.idx_train + self.predict_length > self.cold_start_endpoint):
                raise ValueError("Cold-start active candidate has an unseen future")
        if self.source_candidate_count != len(self.idx_train) + len(self.pending_idx_train):
            raise ValueError("Cold-start candidate count is inconsistent")

    @property
    def pending_count(self) -> int:
        return int(len(self.pending_idx_train))

    def _rebuild_indices(self) -> None:
        self._train_positions_by_tod = {
            int(tod): np.flatnonzero(self.train_tod == tod)
            for tod in np.unique(self.train_tod)
        }
        self._global_mean, self._tod_mean = self._build_fallback_means()

    def _observe_and_activate(self, endpoint: int, query_history: np.ndarray,
                              query_tod: int | None = None) -> None:
        """Append newly observed stream frames and release completed candidates."""
        endpoint = int(endpoint)
        history = np.asarray(query_history, dtype=np.float32)
        expected = (self.input_length, len(self.node_ids))
        if history.shape != expected:
            raise ValueError(f"One query history must be {expected}, got {history.shape}")
        frame_ids = endpoint + self.x_offsets
        observed_through = len(self.train_raw) - 1
        overlap = frame_ids <= observed_through
        if np.any(overlap):
            known = self.train_raw[frame_ids[overlap]]
            if not np.array_equal(known, history[overlap]):
                raise ValueError("Live history disagrees with cold-start snapshot")
        missing = frame_ids > observed_through
        if np.any(missing):
            missing_ids = frame_ids[missing]
            expected_ids = np.arange(observed_through + 1, endpoint + 1)
            if not np.array_equal(missing_ids, expected_ids):
                raise ValueError("Live query stream has a gap")
            self.train_raw = np.concatenate(
                (self.train_raw, history[missing]), axis=0)
        if not len(self.pending_idx_train):
            return
        ready = self.pending_idx_train + self.predict_length <= endpoint
        if not np.any(ready):
            return
        # A ready candidate cannot affect a same-TOD query until its TOD is
        # requested. Delay harmless releases so consecutive boundary candidates
        # are activated together instead of rebuilding fallback statistics once
        # per frame. If a ready TOD is needed now, release it immediately.
        if (not np.all(ready) and query_tod is not None and
                not np.any(self.pending_train_tod[ready] == int(query_tod))):
            return
        ready_endpoints = self.pending_idx_train[ready]
        ready_tod = self.pending_train_tod[ready]
        self.idx_train = np.concatenate((self.idx_train, ready_endpoints))
        self.train_tod = np.concatenate((self.train_tod, ready_tod))
        self.pending_idx_train = self.pending_idx_train[~ready]
        self.pending_train_tod = self.pending_train_tod[~ready]
        order = np.argsort(self.idx_train, kind="stable")
        self.idx_train = self.idx_train[order]
        self.train_tod = self.train_tod[order]
        self._train_positions_by_tod = {
            int(tod): np.flatnonzero(self.train_tod == tod)
            for tod in np.unique(self.train_tod)
        }
        self._add_fallback_candidates(ready_endpoints, ready_tod)

    def _build_fallback_means(self) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        # Accumulate one TOD bucket at a time. Materializing every candidate
        # future at once is unnecessary and causes a large transient RSS spike
        # for PEMS07 full-node cold starts.
        shape = (self.predict_length, len(self.node_ids))
        global_sum = np.zeros(shape, dtype=np.float64)
        global_count = np.zeros(shape, dtype=np.int64)
        tod_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for tod, positions in self._train_positions_by_tod.items():
            endpoints = self.idx_train[positions]
            future = self.train_raw[
                endpoints[:, None] + self.y_offsets[None, :]]
            valid = future != 0
            total = np.where(valid, future, 0.0).sum(axis=0, dtype=np.float64)
            count = valid.sum(axis=0)
            tod_stats[tod] = (total, count)
            global_sum += total
            global_count += count
        self._global_sum = global_sum
        self._global_count = global_count
        self._tod_sum = {tod: total for tod, (total, _) in tod_stats.items()}
        self._tod_count = {tod: count for tod, (_, count) in tod_stats.items()}
        return self._refresh_fallback_means()

    def _refresh_fallback_means(self) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        shape = (self.predict_length, len(self.node_ids))
        global_mean = np.zeros(shape, dtype=np.float64)
        np.divide(self._global_sum, self._global_count, out=global_mean,
                  where=self._global_count > 0)
        if np.any(self._global_count == 0):
            raise ValueError("Train state has an empty global fallback")
        tod_mean: dict[int, np.ndarray] = {}
        for tod, total in self._tod_sum.items():
            count = self._tod_count[tod]
            mean = np.zeros(shape, dtype=np.float64)
            np.divide(total, count, out=mean, where=count > 0)
            tod_mean[tod] = np.where(count > 0, mean, global_mean).T
        return global_mean, tod_mean

    def _add_fallback_candidates(self, endpoints: np.ndarray,
                                 tod_values: np.ndarray) -> None:
        """Increment fallback statistics for newly observable candidates."""
        future = self.train_raw[
            endpoints[:, None] + self.y_offsets[None, :]]
        valid = future != 0
        self._global_sum += np.where(valid, future, 0.0).sum(
            axis=0, dtype=np.float64)
        self._global_count += valid.sum(axis=0)
        shape = self._global_sum.shape
        for tod in np.unique(tod_values):
            selected = future[tod_values == tod]
            selected_valid = selected != 0
            total = np.where(selected_valid, selected, 0.0).sum(
                axis=0, dtype=np.float64)
            count = selected_valid.sum(axis=0)
            key = int(tod)
            if key not in self._tod_sum:
                self._tod_sum[key] = np.zeros(shape, dtype=np.float64)
                self._tod_count[key] = np.zeros(shape, dtype=np.int64)
            self._tod_sum[key] += total
            self._tod_count[key] += count
        self._global_mean, self._tod_mean = self._refresh_fallback_means()

    def validate_causality(self, query_endpoints: np.ndarray,
                           query_tod: np.ndarray) -> None:
        query_endpoints = np.asarray(query_endpoints, dtype=np.int64)
        query_tod = np.asarray(query_tod, dtype=np.int16)
        if query_endpoints.shape != query_tod.shape:
            raise ValueError("Query endpoint/time arrays differ")
        for tod in np.unique(query_tod):
            train_pos = self._train_positions_by_tod.get(int(tod))
            if train_pos is None or len(train_pos) < self.k:
                raise ValueError("Not enough Train candidates for a query calendar bucket")
            candidate_endpoints = self.idx_train[train_pos]
            current_queries = query_endpoints[query_tod == tod]
            if candidate_endpoints.max() + self.predict_length > current_queries.min():
                raise ValueError("A Train candidate future is not observable at query time")

    def query(self, query_history: np.ndarray, query_tod: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return dequantized retrieval, exact value, availability, and INT8 codes."""
        query_history = np.asarray(query_history, dtype=np.float32)
        query_tod = np.asarray(query_tod, dtype=np.int16)
        expected = (len(query_tod), self.input_length, len(self.node_ids))
        if query_history.shape != expected:
            raise ValueError(f"Query history must be {expected}, got {query_history.shape}")
        batch, nodes = len(query_tod), len(self.node_ids)
        raw_retrieval = np.empty(
            (batch, self.predict_length, nodes, 1), dtype=np.float32)
        raw_exact = np.zeros_like(raw_retrieval)
        availability = np.zeros(raw_retrieval.shape, dtype=np.uint8)
        for tod in np.unique(query_tod):
            query_positions = np.flatnonzero(query_tod == tod)
            train_positions = self._train_positions_by_tod.get(int(tod))
            if train_positions is None or len(train_positions) < self.k:
                raise ValueError("Not enough Train candidates for a query calendar bucket")
            endpoints = self.idx_train[train_positions]
            candidate_history = self.train_raw[
                endpoints[:, None] + self.x_offsets[None, :]]
            queries = query_history[query_positions]
            distance = masked_history_distance(
                queries, candidate_history, self.min_valid_history,
                metric="masked_mse")
            nearest = np.argpartition(distance, kth=self.k - 1, axis=1)[:, :self.k, :]
            nearest_distance = np.take_along_axis(distance, nearest, axis=1)
            candidate_future = self.train_raw[
                endpoints[:, None] + self.y_offsets[None, :]].transpose(0, 2, 1)
            node_index = np.arange(nodes)[None, None, :]
            selected_future = candidate_future[nearest, node_index, :]
            selected_valid = np.isfinite(nearest_distance)[..., None] & (selected_future != 0)
            selected_count = selected_valid.sum(axis=1)
            selected_sum = np.where(selected_valid, selected_future, 0.0).sum(
                axis=1, dtype=np.float64)
            retrieved = np.zeros_like(selected_sum, dtype=np.float64)
            np.divide(selected_sum, selected_count, out=retrieved,
                      where=selected_count > 0)
            fallback = self._tod_mean[int(tod)]
            retrieved = np.where(selected_count > 0, retrieved, fallback[None, :, :])
            raw_retrieval[query_positions, :, :, 0] = retrieved.transpose(0, 2, 1)

            query_expanded = queries[:, None, :, :]
            candidate_expanded = candidate_history[None, :, :, :]
            valid = (query_expanded != 0) & (candidate_expanded != 0)
            diff = np.where(valid, query_expanded - candidate_expanded, 0.0)
            squared = np.einsum("qctn,qctn->qcn", diff, diff, optimize=True)
            exact = ((valid.sum(axis=2) >= self.exact_min_valid_history) &
                     (squared == 0))
            exact_valid = exact[..., None] & (candidate_future[None, :, :, :] != 0)
            exact_count = exact_valid.sum(axis=1)
            exact_sum = np.where(exact_valid, candidate_future[None, :, :, :], 0.0).sum(
                axis=1, dtype=np.float64)
            exact_result = np.zeros_like(exact_sum, dtype=np.float64)
            np.divide(exact_sum, exact_count, out=exact_result, where=exact_count > 0)
            raw_exact[query_positions, :, :, 0] = exact_result.transpose(0, 2, 1)
            availability[query_positions, :, :, 0] = (
                exact_count > 0).transpose(0, 2, 1).astype(np.uint8)

        if self.scaler_mean.size == 1:
            mean = float(self.scaler_mean.reshape(-1)[0])
            std = float(self.scaler_std.reshape(-1)[0])
        else:
            mean = self.scaler_mean.reshape(1, 1, -1, 1)
            std = self.scaler_std.reshape(1, 1, -1, 1)
        normalized = ((raw_retrieval - mean) / std).astype(np.float32)
        exact_value = ((raw_exact - mean) / std).astype(np.float32)
        if self.retrieval_format == "float32":
            return normalized, exact_value, availability, normalized
        exact_value = exact_value.astype(np.float16)
        quantized = np.clip(
            np.rint(normalized / self.quant_scale[None, ...]), -127, 127).astype(np.int8)
        dequantized = quantized.astype(np.float32) * self.quant_scale[None, ...]
        return dequantized, exact_value, availability, quantized

    def query_stream(self, query_history: np.ndarray,
                     query_endpoints: np.ndarray, query_tod: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Query an ordered live stream while causally releasing pending candidates."""
        histories = np.asarray(query_history, dtype=np.float32)
        endpoints = np.asarray(query_endpoints, dtype=np.int64)
        tod = np.asarray(query_tod, dtype=np.int16)
        expected = (len(endpoints), self.input_length, len(self.node_ids))
        if histories.shape != expected or tod.shape != endpoints.shape:
            raise ValueError("Streaming query arrays are not aligned")
        if len(endpoints) and np.any(np.diff(endpoints) <= 0):
            raise ValueError("Streaming query endpoints must be strictly increasing")
        outputs = [[], [], [], []]
        cursor = 0
        while cursor < len(endpoints) and self.pending_count:
            self._observe_and_activate(endpoints[cursor], histories[cursor],
                                       int(tod[cursor]))
            current = self.query(histories[cursor:cursor + 1], tod[cursor:cursor + 1])
            for target, value in zip(outputs, current):
                target.append(value)
            cursor += 1
        if cursor < len(endpoints):
            # Once every boundary candidate is active, later stream frames are
            # not part of the frozen Train memory and need not be retained.
            current = self.query(histories[cursor:], tod[cursor:])
            for target, value in zip(outputs, current):
                target.append(value)
        return tuple(np.concatenate(parts, axis=0) for parts in outputs)
