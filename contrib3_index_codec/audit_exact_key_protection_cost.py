"""Measure the real byte cost of protecting exact-history decisions.

This is a storage feasibility audit, not a forecasting result.  It compares
the existing mixed-Q8 container with two additions:

1. an exact uint16 key plane for every history value touched by a candidate;
2. a 64-bit history fingerprint per candidate and selected node.

The first representation is deterministic and collision-free.  The second is
only a practical lower-cost proxy: it can preserve equality decisions with a
negligible collision probability, but it cannot reconstruct exact keys or
exact candidate future values and therefore is never labelled lossless.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SOURCES = {
    "PEMS03": ROOT / "pems03_int8_uniform/sparse_online_state.npz",
    "PEMS04": ROOT / "paper_results/icde_retrieval_free_pilot/online_state_budget_pems04_s1_cap2905847/online_train_state.npz",
    "PEMS07": ROOT / "paper_results/icde_retrieval_free_pilot/pems07_online_state_audit_s1/robust_train_state.npz",
    "PEMS08": ROOT / "paper_results/icde_retrieval_free_pilot/online_state_budget_pems08_s1_cap2785289/online_train_state.npz",
}


def load_codec_module():
    path = HERE / "eval_online_srm_codec.py"
    old_argv = sys.argv
    try:
        sys.argv = [str(path), "PEMS08", "--validation-only"]
        spec = importlib.util.spec_from_file_location("online_codec_eval", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


def archive_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_archive(path: Path, state: dict[str, np.ndarray]) -> int:
    np.savez_compressed(path, **state)
    return path.stat().st_size


def history_positions(idx_train: np.ndarray, input_length: int) -> np.ndarray:
    offsets = np.arange(1 - input_length, 1, dtype=np.int64)
    return np.unique((idx_train[:, None] + offsets[None, :]).reshape(-1))


def fingerprints(raw: np.ndarray, idx_train: np.ndarray, input_length: int,
                 nodes: np.ndarray) -> np.ndarray:
    offsets = np.arange(1 - input_length, 1, dtype=np.int64)
    result = np.empty((len(idx_train), len(nodes)), dtype=np.uint64)
    for row, endpoint in enumerate(idx_train):
        history = np.ascontiguousarray(raw[endpoint + offsets][:, nodes].T,
                                       dtype=np.uint16)
        for column in range(len(nodes)):
            digest = hashlib.blake2b(history[column].tobytes(), digest_size=8).digest()
            result[row, column] = int.from_bytes(digest, "little")
    return result


def audit(dataset: str, nodes: list[int], output: Path,
          source_override: Path | None = None) -> dict:
    source = source_override or SOURCES[dataset]
    if not source.exists():
        raise FileNotFoundError(source)
    codec = load_codec_module()
    nodes_array = np.asarray(sorted(set(nodes)), dtype=np.int32)
    with np.load(source, allow_pickle=False) as archive:
        raw = np.asarray(archive["train_raw"], dtype=np.float32)
        idx_train = np.asarray(archive["idx_train"], dtype=np.int64)
        input_length = int(np.asarray(archive["input_length"]).item())
    if (not len(nodes_array) or np.any(nodes_array < 0) or
            np.any(nodes_array >= raw.shape[1])):
        raise ValueError("Invalid protected node list")
    if not np.array_equal(raw, np.rint(raw)):
        raise ValueError("Exact-key audit requires integral raw traffic values")

    with tempfile.TemporaryDirectory(prefix="exact_key_cost_") as directory:
        temporary = Path(directory)
        lossless_path = temporary / "lossless.npz"
        q8_path = temporary / "q8.npz"
        lossless_bytes = codec.make_mixed_state(
            source, lossless_path, [], pack_index=True)
        q8_bytes = codec.make_mixed_state(
            source, q8_path, nodes_array.tolist(), pack_index=True)
        q8_state = archive_dict(q8_path)

        positions = history_positions(idx_train, input_length)
        exact_state = dict(q8_state)
        exact_state["protected_key_positions"] = positions.astype(np.int32)
        exact_state["protected_key_nodes"] = nodes_array
        exact_state["protected_key_values"] = np.rint(
            raw[positions[:, None], nodes_array[None, :]]).astype(np.uint16)
        exact_path = temporary / "q8_exact_keys.npz"
        exact_bytes = save_archive(exact_path, exact_state)
        with np.load(exact_path, allow_pickle=False) as restored:
            if not np.array_equal(
                    restored["protected_key_values"],
                    np.rint(raw[positions[:, None], nodes_array[None, :]]).astype(np.uint16)):
                raise ValueError("Exact key plane failed roundtrip")

        fingerprint_state = dict(q8_state)
        fingerprint_state["history_fingerprint_nodes"] = nodes_array
        fingerprint_state["history_fingerprint64"] = fingerprints(
            np.rint(raw).astype(np.uint16), idx_train, input_length, nodes_array)
        fingerprint_path = temporary / "q8_fingerprint64.npz"
        fingerprint_bytes = save_archive(fingerprint_path, fingerprint_state)

    result = {
        "dataset": dataset,
        "scope": "storage feasibility only; no model inference; no Test labels",
        "source": str(source),
        "protected_nodes": nodes_array.tolist(),
        "train_raw_shape": list(raw.shape),
        "candidate_count": int(len(idx_train)),
        "input_length": input_length,
        "candidate_history_unique_positions": int(len(positions)),
        "candidate_history_coverage_fraction": float(len(positions) / len(raw)),
        "bytes": {
            "strong_lossless": lossless_bytes,
            "mixed_q8": q8_bytes,
            "mixed_q8_plus_exact_key_plane": exact_bytes,
            "mixed_q8_plus_fingerprint64": fingerprint_bytes,
        },
        "delta_vs_strong_lossless": {
            "mixed_q8": q8_bytes - lossless_bytes,
            "exact_key_plane": exact_bytes - lossless_bytes,
            "fingerprint64": fingerprint_bytes - lossless_bytes,
        },
        "delta_vs_mixed_q8": {
            "exact_key_plane": exact_bytes - q8_bytes,
            "fingerprint64": fingerprint_bytes - q8_bytes,
        },
        "interpretation": {
            "exact_key_plane": "collision-free exact-history key protection; future values remain Q8",
            "fingerprint64": "probabilistic equality proxy only; not independently lossless and does not restore exact future values",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(SOURCES))
    parser.add_argument("--nodes", type=int, nargs="+", required=True)
    parser.add_argument("--source", type=Path,
                        help="Override the default frozen online-state snapshot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset, args.nodes, args.output, args.source), indent=2))


if __name__ == "__main__":
    main()
