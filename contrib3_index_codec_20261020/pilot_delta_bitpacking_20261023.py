"""Exact per-node bit packing of first-order temporal residuals."""
from __future__ import annotations

import importlib.util
import io
import json
import lzma
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"
BASE_REPORT = ONLINE / "reversible_predictive_coding_pilot_20261023.json"
OUT = ONLINE / "delta_bitpacking_pilot_20261023.json"


def load_codec():
    spec = importlib.util.spec_from_file_location(
        "reversible_predictive_pilot", HERE / "pilot_reversible_predictive_coding.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


codec = load_codec()


def pack(values: np.ndarray, bits: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.uint16).reshape(-1).astype(np.uint64)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    matrix = ((flat[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(matrix.reshape(-1))


def unpack(data: np.ndarray, count: int, bits: int) -> np.ndarray:
    bit_values = np.unpackbits(np.asarray(data, dtype=np.uint8))[:count * bits]
    matrix = bit_values.reshape(count, bits).astype(np.uint64)
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint64)
    return (matrix @ weights).astype(np.uint16)


def encode(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = codec.zigzag(raw[1:].astype(np.int32) - raw[:-1].astype(np.int32))
    widths = np.maximum(1, np.ceil(np.log2(codes.max(axis=0).astype(np.float64) + 1))).astype(np.uint8)
    chunks = [pack(codes[:, node], int(widths[node])) for node in range(raw.shape[1])]
    return np.concatenate(chunks), widths, raw[:1].copy()


def decode(stream: np.ndarray, widths: np.ndarray, prefix: np.ndarray,
           n_time: int) -> np.ndarray:
    n_nodes = len(widths)
    codes = np.empty((n_time - 1, n_nodes), dtype=np.uint16)
    cursor = 0
    for node, width_value in enumerate(widths.tolist()):
        width = int(width_value)
        byte_count = ((n_time - 1) * width + 7) // 8
        codes[:, node] = unpack(stream[cursor:cursor + byte_count], n_time - 1, width)
        cursor += byte_count
    if cursor != len(stream):
        raise AssertionError("packed stream length mismatch")
    residuals = codec.unzigzag(codes)
    raw = np.empty((n_time, n_nodes), dtype=np.uint16)
    raw[0] = prefix[0]
    for t in range(1, n_time):
        raw[t] = (raw[t - 1].astype(np.int32) + residuals[t - 1]).astype(np.uint16)
    return raw


def run(dataset: str, base_row: dict) -> dict:
    state, raw = codec.normalize_state(ONLINE / f"{dataset}_s1_val_u16.npz")
    stream, widths, prefix = encode(raw)
    packed_state = dict(state)
    packed_state.pop("train_raw")
    packed_state["train_raw_delta_prefix"] = prefix
    packed_state["train_raw_delta_bits"] = stream
    packed_state["train_raw_delta_widths"] = widths
    blob, order = codec.best_bytes(packed_state)
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    restored = decode(stored["train_raw_delta_bits"],
                      stored["train_raw_delta_widths"],
                      stored["train_raw_delta_prefix"], len(raw))
    if not np.array_equal(restored, raw):
        raise AssertionError(f"{dataset}: bitpacked delta failed exact roundtrip")
    for key, value in state.items():
        if key != "train_raw" and not np.array_equal(stored[key], value):
            raise AssertionError(f"{dataset}: changed field {key}")
    baseline_npz = int(base_row["baseline"]["npz_bytes"])
    baseline_xz = int(base_row["baseline"]["npz_xz_bytes"])
    delta_npz = int(base_row["fixed_lags"]["1"]["npz_bytes"])
    delta_xz = int(base_row["fixed_lags"]["1"]["npz_xz_bytes"])
    return {
        "dataset": dataset.upper(),
        "baseline_npz_bytes": baseline_npz,
        "baseline_npz_xz_bytes": baseline_xz,
        "fixed_delta_npz_bytes": delta_npz,
        "fixed_delta_npz_xz_bytes": delta_xz,
        "per_node_width_bits": widths.astype(int).tolist(),
        "bitpacked_delta_npz_bytes": len(blob),
        "bitpacked_delta_npz_xz_bytes": len(lzma.compress(blob, preset=6)),
        "saved_vs_baseline_npz_bytes": baseline_npz - len(blob),
        "saved_vs_fixed_delta_npz_bytes": delta_npz - len(blob),
        "saved_vs_baseline_xz_bytes": baseline_xz - len(lzma.compress(blob, preset=6)),
        "saved_vs_fixed_delta_xz_bytes": delta_xz - len(lzma.compress(blob, preset=6)),
        "exact_roundtrip_all_fields": True,
        "best_array_order": order,
    }


def main() -> None:
    base = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    rows = {r["dataset"].lower(): r for r in base["results"]}
    result = {
        "title": "Per-node bitpacked exact first-difference pilot",
        "date": "2026-10-23",
        "scope": "s1 Validation online state only; no labels or Test used",
        "protocol": "First-order integer residuals are zig-zag mapped, then bitpacked at each node's observed exact minimum width; complete NPZ and NPZ+xz sizes are measured.",
        "results": [run(ds, rows[ds]) for ds in ("pems03", "pems04", "pems07", "pems08")],
        "not_evaluated": ["Test", "runtime decode latency", "production loader"],
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
