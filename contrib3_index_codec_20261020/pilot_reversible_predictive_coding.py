"""Exploratory exact predictive coding pilot for online Train-state packages.

Reads only the frozen s1 Validation state archives. It compares lossless uint16
storage with fixed-lag and per-node lag-difference transforms. It writes a new
JSON report and does not modify model/checkpoint/configuration artifacts.
"""
from __future__ import annotations

import io
import json
import lzma
import zlib
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"
DATASETS = ("pems03", "pems04", "pems07", "pems08")
LAGS = (1, 12, 288)
OUT = ONLINE / "reversible_predictive_coding_pilot_20261023.json"


def serialize(state: dict[str, np.ndarray], order: str) -> bytes:
    arrays = {
        key: np.array(value, order=order) if value.ndim > 1 else value
        for key, value in state.items()
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def normalize_state(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    raw = np.asarray(state.pop("train_raw"))
    if (not np.isfinite(raw).all() or not np.array_equal(raw, np.rint(raw))
            or raw.min() < 0 or raw.max() > 65535):
        raise ValueError(f"{path}: train_raw is not uint16-exact")
    raw = raw.astype(np.uint16)
    if "idx_train" in state:
        idx = np.asarray(state.pop("idx_train"), dtype=np.int64)
        if len(idx) and not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
            raise ValueError(f"{path}: idx_train is not contiguous")
        state["idx_train_start"] = np.asarray(idx[0], dtype=np.int32)
        state["idx_train_count"] = np.asarray(len(idx), dtype=np.int32)
    state["train_raw"] = raw
    return state, raw


def best_bytes(state: dict[str, np.ndarray]) -> tuple[bytes, str]:
    variants = [(serialize(state, order), order) for order in ("C", "F")]
    return min(variants, key=lambda item: len(item[0]))


def zigzag(diff: np.ndarray) -> np.ndarray:
    diff32 = np.asarray(diff, dtype=np.int32)
    encoded = ((diff32 << 1) ^ (diff32 >> 31)).astype(np.uint32)
    if int(encoded.max(initial=0)) > 65535:
        raise ValueError("difference residual does not fit uint16 zig-zag")
    return encoded.astype(np.uint16)


def unzigzag(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.int32)
    return (value >> 1) ^ -(value & 1)


def fixed_lag_state(state: dict[str, np.ndarray], raw: np.ndarray,
                    lag: int) -> dict[str, np.ndarray]:
    out = dict(state)
    residual = raw.copy()
    residual[lag:] = zigzag(raw[lag:].astype(np.int32) - raw[:-lag].astype(np.int32))
    out.pop("train_raw")
    out["train_raw_prefix"] = raw[:lag]
    out["train_raw_residual_zz"] = residual[lag:]
    out["train_raw_lag"] = np.asarray(lag, dtype=np.uint16)
    return out


def per_node_lag_state(state: dict[str, np.ndarray], raw: np.ndarray
                       ) -> tuple[dict[str, np.ndarray], list[int]]:
    n_time, n_nodes = raw.shape
    lags = np.ones(n_nodes, dtype=np.uint16)
    # Pick each node's lag using compressed residual bytes from that node only.
    # This is an unsupervised, data-only pilot; final choice is then measured on
    # the complete serialized package, including lag metadata and prefixes.
    for node in range(n_nodes):
        choices = []
        for lag in LAGS:
            if lag >= n_time:
                continue
            code = np.concatenate((raw[:lag, node], zigzag(
                raw[lag:, node].astype(np.int32) - raw[:-lag, node].astype(np.int32))))
            choices.append((len(zlib.compress(code.tobytes(), level=9)), lag))
        lags[node] = min(choices)[1]
    residual = np.empty_like(raw)
    for node, lag_value in enumerate(lags.tolist()):
        lag = int(lag_value)
        residual[:lag, node] = raw[:lag, node]
        residual[lag:, node] = zigzag(
            raw[lag:, node].astype(np.int32) - raw[:-lag, node].astype(np.int32))
    out = dict(state)
    out.pop("train_raw")
    out["train_raw_residual_zz"] = residual
    out["train_raw_lags"] = lags
    return out, lags.astype(int).tolist()


def decode_fixed(prefix: np.ndarray, residual: np.ndarray, lag: int) -> np.ndarray:
    decoded = np.empty((len(prefix) + len(residual), residual.shape[1]), dtype=np.uint16)
    decoded[:lag] = prefix
    for t in range(lag, len(decoded)):
        decoded[t] = (decoded[t - lag].astype(np.int32) +
                      unzigzag(residual[t - lag])).astype(np.uint16)
    return decoded


def decode_per_node(residual: np.ndarray, lags: np.ndarray) -> np.ndarray:
    decoded = np.empty_like(residual)
    for node, lag_value in enumerate(lags.tolist()):
        lag = int(lag_value)
        for t in range(lag):
            decoded[t, node] = residual[t, node]
        for t in range(lag, len(decoded)):
            decoded[t, node] = (int(decoded[t - lag, node]) +
                                int(unzigzag(residual[t, node])))
    return decoded


def measure(state: dict[str, np.ndarray]) -> dict:
    blob, layout = best_bytes(state)
    return {"npz_bytes": len(blob), "npz_xz_bytes": len(lzma.compress(blob, preset=6)),
            "best_array_order": layout, "blob": blob}


def verify_archive(blob: bytes, raw: np.ndarray, method: str,
                   expected_state: dict[str, np.ndarray]) -> None:
    """Read the serialized archive back and independently reconstruct raw."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    for key, expected in expected_state.items():
        if key == "train_raw":
            continue
        if key not in stored or not np.array_equal(stored[key], expected):
            raise AssertionError(f"serialized {method} archive changed field {key}")
    if method == "fixed":
        decoded = decode_fixed(stored["train_raw_prefix"],
                               stored["train_raw_residual_zz"],
                               int(stored["train_raw_lag"]))
    elif method == "per_node":
        decoded = decode_per_node(stored["train_raw_residual_zz"],
                                  stored["train_raw_lags"])
    else:
        decoded = stored["train_raw"]
    if not np.array_equal(decoded, raw):
        raise AssertionError(f"serialized {method} archive failed exact roundtrip")


def run_one(dataset: str) -> dict:
    path = ONLINE / f"{dataset}_s1_val_u16.npz"
    state, raw = normalize_state(path)
    # Exact baseline: retain every field; only normalize integral raw histories
    # and contiguous endpoint indices into their lossless compact forms.
    baseline_blob, baseline_order = best_bytes(state)
    verify_archive(baseline_blob, raw, "baseline", state)
    result = {
        "dataset": dataset.upper(),
        "source": path.name,
        "scope": "s1 Validation online state; no labels or Test used",
        "shape": list(raw.shape),
        "raw_min": int(raw.min()),
        "raw_max": int(raw.max()),
        "baseline": {
            "npz_bytes": len(baseline_blob),
            "npz_xz_bytes": len(lzma.compress(baseline_blob, preset=6)),
            "best_array_order": baseline_order,
        },
        "fixed_lags": {},
    }
    for lag in LAGS:
        if lag >= len(raw):
            continue
        encoded = fixed_lag_state(state, raw, lag)
        restored = decode_fixed(encoded["train_raw_prefix"],
                                encoded["train_raw_residual_zz"], lag)
        if not np.array_equal(restored, raw):
            raise AssertionError(f"{dataset}: lag {lag} did not roundtrip")
        measured = measure(encoded)
        verify_archive(measured["blob"], raw, "fixed", state)
        measured.pop("blob")
        result["fixed_lags"][str(lag)] = {
            **measured,
            "saved_vs_baseline_npz_bytes": len(baseline_blob) - measured["npz_bytes"],
            "saved_vs_baseline_xz_bytes": result["baseline"]["npz_xz_bytes"] - measured["npz_xz_bytes"],
            "exact_roundtrip": True,
        }
    encoded, chosen_lags = per_node_lag_state(state, raw)
    restored = decode_per_node(encoded["train_raw_residual_zz"], encoded["train_raw_lags"])
    if not np.array_equal(restored, raw):
        raise AssertionError(f"{dataset}: per-node lag did not roundtrip")
    measured = measure(encoded)
    verify_archive(measured["blob"], raw, "per_node", state)
    measured.pop("blob")
    result["per_node_lag"] = {
        **measured,
        "lags": chosen_lags,
        "saved_vs_baseline_npz_bytes": len(baseline_blob) - measured["npz_bytes"],
        "saved_vs_baseline_xz_bytes": result["baseline"]["npz_xz_bytes"] - measured["npz_xz_bytes"],
        "exact_roundtrip": True,
    }
    return result


def main() -> None:
    results = [run_one(dataset) for dataset in DATASETS]
    report = {
        "title": "Reversible temporal predictive coding pilot",
        "date": "2026-10-23",
        "protocol": {
            "datasets": list(DATASETS),
            "splits": ["s1 Validation only"],
            "candidate_predictors": list(LAGS),
            "selection": "Per-node lag selected using that node's compressed residual bytes; final score is complete serialized NPZ and NPZ+xz size.",
            "losslessness": "Every candidate passed uint16 train_raw elementwise roundtrip before size reporting.",
            "not_evaluated": ["MAE rerun", "Test", "runtime decode latency", "cold-start deployment loader"],
        },
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
