"""Test additional exact predictors not covered by the lag-only pilot."""
from __future__ import annotations

import importlib.util
import io
import json
import lzma
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"
BASE_PILOT = ONLINE / "reversible_predictive_coding_pilot_20261023.json"
OUT = ONLINE / "predictor_family_pilot_20261023.json"
DATASETS = ("pems03", "pems04", "pems07", "pems08")
MODES = {"lag1": 1, "linear2": 2, "seasonal_trend288": 289}


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "reversible_predictive_pilot", HERE / "pilot_reversible_predictive_coding.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


codec = load_base_module()


def predict(raw: np.ndarray, mode: str, t: int) -> np.ndarray:
    if mode == "lag1":
        return raw[t - 1].astype(np.int32)
    if mode == "linear2":
        return 2 * raw[t - 1].astype(np.int32) - raw[t - 2].astype(np.int32)
    if mode == "seasonal_trend288":
        return (raw[t - 288].astype(np.int32) + raw[t - 1].astype(np.int32)
                - raw[t - 289].astype(np.int32))
    raise ValueError(mode)


def encode(raw: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    warm = MODES[mode]
    residual = raw.copy()
    for t in range(warm, len(raw)):
        residual[t] = codec.zigzag(raw[t].astype(np.int32) - predict(raw, mode, t))
    return raw[:warm].copy(), residual[warm:].copy()


def decode(prefix: np.ndarray, residual: np.ndarray, mode: str) -> np.ndarray:
    warm = MODES[mode]
    out = np.empty((len(prefix) + len(residual), residual.shape[1]), dtype=np.uint16)
    out[:warm] = prefix
    for t in range(warm, len(out)):
        if mode == "lag1":
            pred = out[t - 1].astype(np.int32)
        elif mode == "linear2":
            pred = 2 * out[t - 1].astype(np.int32) - out[t - 2].astype(np.int32)
        else:
            pred = (out[t - 288].astype(np.int32) + out[t - 1].astype(np.int32)
                    - out[t - 289].astype(np.int32))
        out[t] = (pred + codec.unzigzag(residual[t - warm])).astype(np.uint16)
    return out


def package_state(state: dict[str, np.ndarray], raw: np.ndarray,
                  mode: str) -> dict[str, np.ndarray]:
    prefix, residual = encode(raw, mode)
    out = dict(state)
    out.pop("train_raw")
    out["train_raw_prefix"] = prefix
    out["train_raw_residual_zz"] = residual
    out["train_raw_predictor"] = np.asarray(list(MODES).index(mode), dtype=np.uint8)
    return out


def measure_and_verify(state: dict[str, np.ndarray], raw: np.ndarray,
                       mode: str, expected: dict[str, np.ndarray]) -> dict:
    candidate = package_state(state, raw, mode)
    blob, order = codec.best_bytes(candidate)
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        loaded = {key: archive[key] for key in archive.files}
    restored = decode(loaded["train_raw_prefix"],
                     loaded["train_raw_residual_zz"], mode)
    if not np.array_equal(restored, raw):
        raise AssertionError(f"{mode}: exact raw roundtrip failed")
    for key, value in expected.items():
        if key != "train_raw" and not np.array_equal(loaded[key], value):
            raise AssertionError(f"{mode}: field changed: {key}")
    return {"npz_bytes": len(blob),
            "npz_xz_bytes": len(lzma.compress(blob, preset=6)),
            "best_array_order": order, "exact_roundtrip": True}


def encode_per_node(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_time, n_nodes = raw.shape
    mode_names = list(MODES)
    selected = np.empty(n_nodes, dtype=np.uint8)
    residual = np.empty_like(raw)
    for node in range(n_nodes):
        choices = []
        for mode_id, mode in enumerate(mode_names):
            prefix, tail = encode(raw[:, node:node + 1], mode)
            # Include seed bytes and one-byte mode id in the local screening cost.
            local = np.concatenate((prefix.reshape(-1), tail.reshape(-1)))
            choices.append((len(zlib.compress(local.tobytes(), 9)) + 1,
                            mode_id, mode, prefix.reshape(-1), tail.reshape(-1)))
        _, mode_id, mode, prefix, tail = min(choices, key=lambda x: (x[0], x[1]))
        warm = MODES[mode]
        selected[node] = mode_id
        residual[:warm, node] = prefix
        residual[warm:, node] = tail
    return residual, selected


def decode_per_node(residual: np.ndarray, selected: np.ndarray) -> np.ndarray:
    out = np.empty_like(residual)
    mode_names = list(MODES)
    for node, mode_id in enumerate(selected.tolist()):
        mode = mode_names[int(mode_id)]
        warm = MODES[mode]
        out[:warm, node] = residual[:warm, node]
        for t in range(warm, len(out)):
            if mode == "lag1":
                pred = int(out[t - 1, node])
            elif mode == "linear2":
                pred = 2 * int(out[t - 1, node]) - int(out[t - 2, node])
            else:
                pred = (int(out[t - 288, node]) + int(out[t - 1, node])
                        - int(out[t - 289, node]))
            out[t, node] = (pred + int(codec.unzigzag(residual[t, node]))) & 0xFFFF
    return out


def run(dataset: str, baseline_row: dict) -> dict:
    path = ONLINE / f"{dataset}_s1_val_u16.npz"
    state, raw = codec.normalize_state(path)
    rows = {}
    for mode in MODES:
        row = measure_and_verify(state, raw, mode, state)
        rows[mode] = row
    # Per-node mode selector: retain an in-place residual matrix and one mode id
    # per sensor; each node's selected predictor is independently round-tripped.
    residual, selected = encode_per_node(raw)
    candidate = dict(state)
    candidate.pop("train_raw")
    candidate["train_raw_residual_zz"] = residual
    candidate["train_raw_predictors"] = selected
    blob, order = codec.best_bytes(candidate)
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        loaded = {key: archive[key] for key in archive.files}
    restored = decode_per_node(loaded["train_raw_residual_zz"],
                               loaded["train_raw_predictors"])
    if not np.array_equal(restored, raw):
        raise AssertionError("per-node predictor selector failed exact roundtrip")
    for key, value in state.items():
        if key != "train_raw" and not np.array_equal(loaded[key], value):
            raise AssertionError(f"per-node selector changed field {key}")
    adaptive = {"npz_bytes": len(blob),
                "npz_xz_bytes": len(lzma.compress(blob, preset=6)),
                "best_array_order": order,
                "selected_modes": {mode: int(np.count_nonzero(selected == i))
                                   for i, mode in enumerate(MODES)},
                "exact_roundtrip": True}
    prior = baseline_row["fixed_lags"]["1"]
    return {
        "dataset": dataset.upper(),
        "baseline_npz_bytes": baseline_row["baseline"]["npz_bytes"],
        "baseline_npz_xz_bytes": baseline_row["baseline"]["npz_xz_bytes"],
        "already_tested_fixed_lag1": prior,
        "new_fixed_predictors": rows,
        "new_per_node_predictor": adaptive,
    }


def main() -> None:
    base_report = json.loads(BASE_PILOT.read_text(encoding="utf-8"))
    indexed = {row["dataset"].lower(): row for row in base_report["results"]}
    results = [run(dataset, indexed[dataset]) for dataset in DATASETS]
    out = {
        "title": "Additional exact predictor-family pilot",
        "date": "2026-10-23",
        "scope": "s1 Validation online states only; no labels or Test used",
        "new_predictors": ["second-order linear extrapolation", "daily seasonal value plus recent trend"],
        "per_node_selector": "Select among lag1, linear2, and seasonal_trend288 using per-node compressed residual bytes; include one-byte mode id in local screening, then measure complete package.",
        "losslessness": "Serialized NPZ readback; raw reconstructed elementwise; all other state fields checked elementwise.",
        "not_evaluated": ["Test", "runtime decode latency", "production cold-start loader"],
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
