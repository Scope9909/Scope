"""Runtime compatibility check for the exact delta-bitpacked state format."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"
OUT = ONLINE / "bitpacked_runtime_query_pilot_20261023.json"
DATASETS = ("pems03", "pems04", "pems07", "pems08")


def module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


codec = module_from_file("delta_bitpack_codec", HERE / "pilot_delta_bitpacking_20261023.py")
base_codec = module_from_file("lossless_state_codec", HERE / "pilot_reversible_predictive_coding.py")
sys.path.insert(0, str(ROOT))
from utils.online_residual_memory import OnlineResidualMemory  # noqa: E402


def package_state(source: Path, destination: Path) -> tuple[int, np.ndarray]:
    state, raw = base_codec.normalize_state(source)
    stream, widths, prefix = codec.encode(raw)
    state.pop("train_raw")
    state["train_raw_delta_prefix"] = prefix
    state["train_raw_delta_bits"] = stream
    state["train_raw_delta_widths"] = widths
    state["train_raw_delta_shape"] = np.asarray(raw.shape, dtype=np.uint32)
    state["train_raw_delta_codec_version"] = np.asarray(1, dtype=np.uint8)
    blob, _ = base_codec.best_bytes(state)
    destination.write_bytes(blob)
    return len(blob), raw


def load_decoded(path: Path) -> tuple[dict[str, np.ndarray], float]:
    start = time.perf_counter()
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    shape = tuple(int(v) for v in state.pop("train_raw_delta_shape"))
    version = int(state.pop("train_raw_delta_codec_version"))
    if version != 1:
        raise ValueError(f"Unsupported delta codec version: {version}")
    raw = codec.decode(state.pop("train_raw_delta_bits"),
                       state.pop("train_raw_delta_widths"),
                       state.pop("train_raw_delta_prefix"), shape[0])
    if raw.shape != shape:
        raise ValueError("Decoded raw history has unexpected shape")
    state["train_raw"] = raw.astype(np.float32)
    if "idx_train_start" in state:
        start_index = int(np.asarray(state.pop("idx_train_start")).item())
        count = int(np.asarray(state.pop("idx_train_count")).item())
        if count <= 0:
            raise ValueError("Invalid packed candidate-index count")
        state["idx_train"] = np.arange(start_index, start_index + count, dtype=np.int32)
    return state, time.perf_counter() - start


def query_check(dataset: str) -> dict:
    source = ONLINE / f"{dataset}_s1_val_u16.npz"
    compact_state, raw = base_codec.normalize_state(source)
    optimized_blob, _ = base_codec.best_bytes(compact_state)
    state = dict(compact_state)
    start_index = int(np.asarray(state.pop("idx_train_start")).item())
    count = int(np.asarray(state.pop("idx_train_count")).item())
    state["idx_train"] = np.arange(start_index, start_index + count, dtype=np.int32)
    state["train_raw"] = raw.astype(np.float32)
    start = time.perf_counter()
    baseline_memory = OnlineResidualMemory(state)
    baseline_init_seconds = time.perf_counter() - start

    with tempfile.TemporaryDirectory(prefix=f"{dataset}_bitpacked_", dir=ONLINE) as td:
        packed_path = Path(td) / "candidate.npz"
        encoded_bytes, exact_raw = package_state(source, packed_path)
        decoded_state, decode_seconds = load_decoded(packed_path)
        if not np.array_equal(decoded_state["train_raw"], exact_raw.astype(np.float32)):
            raise AssertionError("Serialized package does not reconstruct exact raw history")
        for key, value in state.items():
            if key != "train_raw" and not np.array_equal(decoded_state[key], value):
                raise AssertionError(f"Runtime package changed field {key}")
        start = time.perf_counter()
        decoded_memory = OnlineResidualMemory(decoded_state)
        decoded_init_seconds = time.perf_counter() - start

        rng = np.random.default_rng(20261023)
        candidates = rng.choice(len(state["idx_train"]), size=min(4, len(state["idx_train"])),
                                replace=False)
        endpoints = np.asarray(state["idx_train"], dtype=np.int64)[candidates]
        offsets = np.arange(1 - baseline_memory.input_length, 1, dtype=np.int64)
        query = exact_raw[endpoints[:, None] + offsets[None, :]].astype(np.float32)
        tod = np.asarray(state["train_tod"])[candidates].astype(np.int16)
        # Exercise zero-masked equality and non-exact nearest retrieval as well.
        if len(query) > 1:
            query[1, 0, 0] = 0
        if len(query) > 2:
            query[2, 0, min(1, raw.shape[1] - 1)] += 1

        start = time.perf_counter()
        baseline_outputs = baseline_memory.query(query, tod)
        baseline_query_seconds = time.perf_counter() - start
        start = time.perf_counter()
        decoded_outputs = decoded_memory.query(query, tod)
        decoded_query_seconds = time.perf_counter() - start
        output_exact = all(np.array_equal(a, b) for a, b in zip(baseline_outputs, decoded_outputs))
        if not output_exact:
            raise AssertionError(f"{dataset}: retrieval outputs changed after decoding")
        return {
            "dataset": dataset.upper(),
            "source_npz_bytes": source.stat().st_size,
            "strong_lossless_same_layout_npz_bytes": len(optimized_blob),
            "bitpacked_npz_bytes": encoded_bytes,
            "saved_bytes": len(optimized_blob) - encoded_bytes,
            "saved_fraction": 1 - encoded_bytes / len(optimized_blob),
            "query_count": len(query),
            "query_cases": ["exact", "zero-masked", "one-value-perturbed", "exact"],
            "raw_elementwise_exact": True,
            "all_other_fields_exact": True,
            "retrieval_outputs_bitwise_equal": output_exact,
            "baseline_memory_init_seconds": baseline_init_seconds,
            "packed_decode_seconds": decode_seconds,
            "decoded_memory_init_seconds": decoded_init_seconds,
            "baseline_query_seconds": baseline_query_seconds,
            "decoded_query_seconds": decoded_query_seconds,
        }


def main() -> None:
    rows = [query_check(dataset) for dataset in DATASETS]
    result = {
        "title": "Delta-bitpacked state runtime query compatibility pilot",
        "date": "2026-10-23",
        "scope": "s1 Validation state archives; no labels or Test used",
        "query": "Actual OnlineResidualMemory.query; four candidate-derived query histories per dataset, including zero masking and a one-value perturbation.",
        "results": rows,
        "limitations": ["No full Student prediction/MAE rerun", "Small query sample; latency is diagnostic only", "Temporary NPZ candidate is deleted after each dataset"],
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
