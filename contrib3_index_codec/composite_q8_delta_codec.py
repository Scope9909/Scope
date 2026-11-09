"""Validation-only pilot: lossless delta-bitpacking of Q8 mixed histories.

The existing mixed-state Q8 representation contains two integer streams:
uint8 Q-codes and uint16 exact-node values. This wrapper delta-bitpacks each
stream independently, while retaining all Q8 scales, node maps, and tails.
It is intentionally independent of Test data and does not change the runtime.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np

from delta_bitpacked_state_codec import encode_state, decode_raw, serialize
from online_codec_runtime import decode_online_state


ROOT = Path(__file__).resolve().parent
ONLINE = ROOT / "packages" / "online"
CASES = {
    "pems03": "pems03_s1_val_fixed_22_idx.npz",
    "pems04": "pems04_s1_val_fixed_0_idx.npz",
    "pems07": "pems07_s1_val_fixed_117_idx.npz",
    "pems08": "pems08_s1_val_fixed_zero_idx.npz",
}
LOSSLESS_PACKAGES = {
    "pems03": ("pems03_s1_val_delta_bitpacked.npz", 275907),
    "pems04": ("pems04_s1_val_delta_bitpacked.npz", 79579),
    "pems07": ("pems07_s1_val_delta_bitpacked.npz", 356299),
    "pems08": ("pems08_s1_val_delta_bitpacked.npz", 33803),
}
STREAMS = ("train_raw_mixed_q_codes", "train_raw_mixed_exact")


def _encode_stream(raw: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    # encode_state accepts nonnegative integer matrices up to uint16. Q-codes
    # are uint8 and exact-node history is uint16, so both are valid inputs.
    if raw.shape[1] == 0:
        return {f"{prefix}_shape": np.asarray(raw.shape, dtype=np.uint32),
                f"{prefix}_empty": np.asarray(1, dtype=np.uint8)}
    encoded = encode_state({"train_raw": raw})
    return {f"{prefix}_{key.removeprefix('train_raw_delta_')}": value
            for key, value in encoded.items() if key != "train_raw_delta_codec_id"}


def _decode_stream(state: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    if int(state.get(f"{prefix}_empty", np.asarray(0))):
        return np.empty(tuple(map(int, state[f"{prefix}_shape"])), dtype=np.uint16)
    return decode_raw(
        state[f"{prefix}_prefix"], state[f"{prefix}_bits"],
        state[f"{prefix}_widths"], tuple(map(int, state[f"{prefix}_shape"])),
        state[f"{prefix}_sha256"],
    )


def pack_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v) for k, v in state.items() if k not in STREAMS}
    for key in STREAMS:
        if key not in state:
            raise ValueError(f"Missing mixed-state stream: {key}")
        raw = np.asarray(state[key])
        prefix = "qcodes" if key.endswith("q_codes") else "exact"
        encoded = _encode_stream(raw, prefix)
        out.update(encoded)
    out["mixed_delta_codec_version"] = np.asarray(1, dtype=np.uint8)
    return out


def unpack_state(packed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v) for k, v in packed.items()
           if k not in {"mixed_delta_codec_version", *(f"{p}_{s}" for p in ("qcodes", "exact")
                                                       for s in ("empty", "shape", "prefix", "widths", "bits", "sha256"))}}
    out["train_raw_mixed_q_codes"] = _decode_stream(packed, "qcodes").astype(np.uint8)
    out["train_raw_mixed_exact"] = _decode_stream(packed, "exact").astype(np.uint16)
    return out


def decode_composite_online_state(path: str | Path) -> dict[str, np.ndarray]:
    """Decode a composite package through the production online-state decoder."""
    with np.load(path, allow_pickle=False) as archive:
        packed = {key: archive[key] for key in archive.files}
    restored = unpack_state(packed)
    with tempfile.TemporaryDirectory(dir=Path(path).resolve().parent) as temp_dir:
        intermediate = Path(temp_dir) / "restored_q8_state.npz"
        np.savez_compressed(intermediate, **restored)
        return decode_online_state(intermediate)


def _serialize(state: dict[str, np.ndarray]) -> tuple[bytes, str]:
    candidates = []
    for order in ("C", "F"):
        arrays = {k: np.array(v, order=order) if v.ndim > 1 else v for k, v in state.items()}
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        candidates.append((stream.getvalue(), order))
    return min(candidates, key=lambda item: len(item[0]))


def run() -> dict:
    results = {}
    for dataset, filename in CASES.items():
        source = ONLINE / filename
        target = ONLINE / f"{dataset}_s1_val_q8_delta_pilot.npz"
        with np.load(source, allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        packed = pack_state(original)
        blob, order = _serialize(packed)
        target.write_bytes(blob)
        restored = unpack_state(packed)
        mismatches = [key for key in original
                      if key not in restored or not np.array_equal(original[key], restored[key])]
        if mismatches:
            raise AssertionError(f"Round-trip mismatch for {dataset}: {mismatches}")
        base_runtime = decode_online_state(source)
        composite_runtime = decode_composite_online_state(target)
        runtime_mismatches = [key for key in base_runtime
                              if key not in composite_runtime or
                              not np.array_equal(base_runtime[key], composite_runtime[key])]
        if runtime_mismatches:
            raise AssertionError(f"Runtime decode mismatch for {dataset}: {runtime_mismatches}")
        lossless_name, strong_lossless_bytes = LOSSLESS_PACKAGES[dataset]
        lossless_packed_path = ONLINE / "bitpacked_val" / lossless_name
        lossless_packed_bytes = lossless_packed_path.stat().st_size
        composite_bytes = len(blob)
        if composite_bytes < lossless_packed_bytes:
            selected_codec = "Q8 + exact delta-bitpacking"
            selected_bytes = composite_bytes
        else:
            selected_codec = "strong-lossless + exact delta-bitpacking"
            selected_bytes = lossless_packed_bytes
        results[dataset] = {
            "validation_source": filename,
            "q8_nodes": original["train_raw_mixed_q_nodes"].astype(int).tolist(),
            "baseline_q8_bytes": source.stat().st_size,
            "composite_bytes": composite_bytes,
            "bytes_saved": source.stat().st_size - len(blob),
            "saved_percent": 100.0 * (source.stat().st_size - len(blob)) / source.stat().st_size,
            "bitpacked_lossless_bytes": lossless_packed_bytes,
            "q8_composite_vs_bitpacked_lossless_bytes": composite_bytes - lossless_packed_bytes,
            "frozen_strong_lossless_bytes": strong_lossless_bytes,
            "validation_cost_selected_codec": selected_codec,
            "validation_cost_selected_bytes": selected_bytes,
            "serialization_order": order,
            "decoded_state_bitwise_equal": True,
            "runtime_decoded_state_bitwise_equal": True,
            "test_data_read": False,
        }
    summary = {"scope": "Validation-only feasibility pilot; no Test data read",
               "results": results,
               "total_baseline_bytes": sum(x["baseline_q8_bytes"] for x in results.values()),
               "total_composite_bytes": sum(x["composite_bytes"] for x in results.values()),
               "total_strong_lossless_bytes": sum(x["frozen_strong_lossless_bytes"] for x in results.values()),
               "total_bitpacked_lossless_bytes": sum(x["bitpacked_lossless_bytes"] for x in results.values()),
               "total_validation_cost_selected_bytes": sum(x["validation_cost_selected_bytes"] for x in results.values())}
    summary["total_saved_bytes"] = summary["total_baseline_bytes"] - summary["total_composite_bytes"]
    summary["total_saved_percent"] = 100.0 * summary["total_saved_bytes"] / summary["total_baseline_bytes"]
    summary["selected_saved_vs_strong_lossless_bytes"] = summary["total_strong_lossless_bytes"] - summary["total_validation_cost_selected_bytes"]
    summary["selected_saved_vs_strong_lossless_percent"] = 100.0 * summary["selected_saved_vs_strong_lossless_bytes"] / summary["total_strong_lossless_bytes"]
    report = ONLINE / "q8_delta_composition_validation_pilot.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
