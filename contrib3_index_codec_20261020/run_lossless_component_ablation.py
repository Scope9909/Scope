"""Validation-only ablation of reversible transforms in the ALSC container.

The compressor family, metadata accounting, source states, and verification
are identical across policies.  This script never loads labels or Test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import adaptive_lossless_state_codec as codec
from pilot_reversible_predictive_coding import normalize_state

HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"
CASES = {
    "pems03": "pems03_s1_val_fixed_22_idx.npz",
    "pems04": None,
    "pems07": "pems07_s1_val_fixed_117_idx.npz",
    "pems08": None,
}
POLICIES = ("direct", "delta_only", "reorder_only", "delta_reorder", "adaptive")


def load_state(path: Path, normalize: bool) -> dict[str, np.ndarray]:
    if normalize:
        state, _ = normalize_state(path)
        return state
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def encode_policy(state: dict[str, np.ndarray], policy: str):
    if policy == "adaptive":
        blob, _, fields = codec.encode(state)
        return blob, fields
    rows, payloads, diagnostics = [], [], []
    allow_delta = policy in {"delta_only", "delta_reorder"}
    allow_reorder = policy in {"reorder_only", "delta_reorder"}
    for key in sorted(state):
        a = np.asarray(state[key])
        history = key in codec.HISTORY_FIELDS and a.ndim == 2 and a.size > 0
        predictors = ("delta",) if history and allow_delta else ("none",)
        layouts = ("plain", "bytes", "planes") if allow_reorder and a.size > 512 else ("plain",)
        candidates = []
        for predictor in predictors:
            for order in (("C", "F") if a.ndim > 1 else ("C",)):
                for layout in layouts:
                    data, encoded_dtype = codec.transform(a, order, predictor, layout)
                    for compression, blob in codec.compressions(data):
                        row = dict(key=key, dtype=a.dtype.str, shape=list(a.shape),
                                   encoded_dtype=encoded_dtype, order=order,
                                   predictor=predictor, layout=layout,
                                   compression=compression, length=len(blob),
                                   sha256=codec.digest(a))
                        candidates.append((len(blob) + len(codec.json_bytes(row)), row, blob))
        best = min(candidates, key=lambda item: item[0])
        restored = codec.restore(best[2], best[1])
        if restored.dtype != a.dtype or restored.shape != a.shape or restored.tobytes() != a.tobytes():
            raise AssertionError(f"Round-trip failed: {key}/{policy}")
        rows.append(best[1])
        payloads.append(best[2])
        diagnostics.append(dict(best[1], original_bytes=a.nbytes))
    return codec.assemble(rows, payloads), diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for dataset, mixed_name in CASES.items():
        exact_path = ONLINE / f"{dataset}_s1_val_u16.npz"
        exact = load_state(exact_path, normalize=True)
        selected_path = ONLINE / mixed_name if mixed_name else exact_path
        selected = load_state(selected_path, normalize=not bool(mixed_name))
        for state_name, state, source in (("exact", exact, exact_path),
                                          ("selected", selected, selected_path)):
            for policy in POLICIES:
                started = time.perf_counter()
                blob, fields = encode_policy(state, policy)
                decoded = codec.decode(blob)
                if decoded.keys() != state.keys():
                    raise AssertionError("Field set changed")
                for key in state:
                    if decoded[key].dtype != state[key].dtype or decoded[key].shape != state[key].shape:
                        raise AssertionError(f"dtype/shape changed: {key}")
                    if decoded[key].tobytes() != state[key].tobytes():
                        raise AssertionError(f"bytes changed: {key}")
                target = args.output / f"{dataset}_val_{state_name}_{policy}.alsc"
                target.write_bytes(blob)
                results.append(dict(dataset=dataset.upper(), split="Validation",
                                    state=state_name, policy=policy, bytes=len(blob),
                                    elapsed_seconds=time.perf_counter() - started,
                                    source=str(source.resolve()),
                                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                    archive_sha256=hashlib.sha256(blob).hexdigest(),
                                    byte_exact_roundtrip=True, fields=fields))
                print(dataset, state_name, policy, len(blob), flush=True)
    totals = {}
    for state_name in ("exact", "selected"):
        totals[state_name] = {}
        for policy in POLICIES:
            total = sum(row["bytes"] for row in results
                        if row["state"] == state_name and row["policy"] == policy)
            totals[state_name][policy] = total
    report = dict(scope="Validation-only reversible component ablation",
                  test_used=False, labels_used=False, q8_nodes_changed=False,
                  policies={
                      "direct": "no delta; plain byte order; best C/F and compressor",
                      "delta_only": "history delta+ZigZag; no byte/bit-plane rearrangement",
                      "reorder_only": "no delta; per-field plain/byte/bit-plane choice",
                      "delta_reorder": "history delta+ZigZag plus per-field rearrangement choice",
                      "adaptive": "per-field choice over direct/delta and plain/byte/bit-plane",
                  }, results=results, totals=totals)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
