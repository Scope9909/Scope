"""Versioned exact delta/bitpacked codec for integer online Train histories."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Mapping

import numpy as np

CODEC_ID = 1


def _zigzag(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.int32)
    encoded = ((value << 1) ^ (value >> 31)).astype(np.uint32)
    if int(encoded.max(initial=0)) > 65535:
        raise ValueError("Residual exceeds uint16 zig-zag range")
    return encoded.astype(np.uint16)


def _unzigzag(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.int32)
    return (value >> 1) ^ -(value & 1)


def _pack(values: np.ndarray, bits: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.uint16).reshape(-1).astype(np.uint64)
    if bits < 1 or bits > 16:
        raise ValueError(f"Invalid bit width {bits}")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    return np.packbits(((flat[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1))


def _unpack(values: np.ndarray, count: int, bits: int) -> np.ndarray:
    raw = np.unpackbits(np.asarray(values, dtype=np.uint8))[:count * bits]
    matrix = raw.reshape(count, bits).astype(np.uint64)
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint64)
    return (matrix @ weights).astype(np.uint16)


def raw_checksum(raw: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(raw, dtype="<u2")
    return np.frombuffer(hashlib.sha256(canonical.tobytes()).digest(), dtype=np.uint8).copy()


def decode_raw(prefix: np.ndarray, bitstream: np.ndarray, widths: np.ndarray,
               shape: tuple[int, int], checksum: np.ndarray) -> np.ndarray:
    n_time, n_nodes = shape
    if n_time < 1 or n_nodes < 1 or prefix.shape != (1, n_nodes):
        raise ValueError("Malformed delta-bitpacked shape/prefix")
    widths = np.asarray(widths, dtype=np.uint8)
    if widths.shape != (n_nodes,) or np.any((widths < 1) | (widths > 16)):
        raise ValueError("Malformed per-node residual widths")
    expected_bytes = sum(((n_time - 1) * int(bits) + 7) // 8 for bits in widths)
    if np.asarray(bitstream).ndim != 1 or len(bitstream) != expected_bytes:
        raise ValueError("Malformed packed residual stream length")
    codes = np.empty((n_time - 1, n_nodes), dtype=np.uint16)
    cursor = 0
    for node, width_value in enumerate(widths.tolist()):
        width = int(width_value)
        size = ((n_time - 1) * width + 7) // 8
        codes[:, node] = _unpack(bitstream[cursor:cursor + size], n_time - 1, width)
        cursor += size
    residuals = _unzigzag(codes)
    raw = np.empty(shape, dtype=np.uint16)
    raw[0] = prefix[0]
    for t in range(1, n_time):
        raw[t] = (raw[t - 1].astype(np.int32) + residuals[t - 1]).astype(np.uint16)
    if not np.array_equal(raw_checksum(raw), np.asarray(checksum, dtype=np.uint8)):
        raise ValueError("Delta-bitpacked Train history checksum mismatch")
    return raw


def encode_state(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "train_raw" not in state:
        raise ValueError("Missing train_raw")
    raw_float = np.asarray(state["train_raw"])
    if (raw_float.ndim != 2 or not np.isfinite(raw_float).all() or
            not np.array_equal(raw_float, np.rint(raw_float)) or
            raw_float.min() < 0 or raw_float.max() > 65535):
        raise ValueError("train_raw must be a 2-D uint16-exact nonnegative array")
    raw = np.ascontiguousarray(raw_float, dtype=np.uint16)
    differences = raw[1:].astype(np.int32) - raw[:-1].astype(np.int32)
    codes = _zigzag(differences)
    maxima = codes.max(axis=0) if len(codes) else np.zeros(raw.shape[1], dtype=np.uint16)
    widths = np.maximum(1, np.ceil(np.log2(maxima.astype(np.float64) + 1))).astype(np.uint8)
    chunks = [_pack(codes[:, node], int(widths[node])) for node in range(raw.shape[1])]
    out = {key: np.asarray(value) for key, value in state.items() if key != "train_raw"}
    if "idx_train" in out:
        indices = np.asarray(out.pop("idx_train"), dtype=np.int64)
        if len(indices) == 0 or not np.array_equal(
                indices, np.arange(indices[0], indices[0] + len(indices))):
            raise ValueError("Only nonempty contiguous idx_train can be compacted")
        out["idx_train_start"] = np.asarray(indices[0], dtype=np.int32)
        out["idx_train_count"] = np.asarray(len(indices), dtype=np.int32)
    out["train_raw_delta_codec_id"] = np.asarray(CODEC_ID, dtype=np.uint8)
    out["train_raw_delta_shape"] = np.asarray(raw.shape, dtype=np.uint32)
    out["train_raw_delta_prefix"] = raw[:1]
    out["train_raw_delta_widths"] = widths
    out["train_raw_delta_bits"] = np.concatenate(chunks) if chunks else np.empty(0, np.uint8)
    out["train_raw_delta_sha256"] = raw_checksum(raw)
    return out


def serialize(state: Mapping[str, np.ndarray]) -> tuple[bytes, str]:
    variants = []
    for order in ("C", "F"):
        arrays = {key: np.array(value, order=order) if value.ndim > 1 else value
                  for key, value in state.items()}
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        variants.append((stream.getvalue(), order))
    return min(variants, key=lambda row: len(row[0]))


def save(source: str | Path, destination: str | Path) -> dict:
    with np.load(source, allow_pickle=False) as archive:
        original = {key: archive[key] for key in archive.files}
    packed = encode_state(original)
    blob, order = serialize(packed)
    Path(destination).write_bytes(blob)
    return {"bytes": len(blob), "array_order": order,
            "raw_shape": list(original["train_raw"].shape),
            "checksum": raw_checksum(original["train_raw"]).tobytes().hex()}
