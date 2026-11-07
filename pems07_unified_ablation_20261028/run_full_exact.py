"""Match PEMS07 all-node exact memory to the frozen SRM codec protocol.

The node set is not reselected. The same Student checkpoint, split, query
implementation, calibration hyperparameters, and field recipes are reused.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CODEC_DIR = ROOT / "contrib3_index_codec_20261020"
sys.path.insert(0, str(CODEC_DIR))
import adaptive_lossless_state_codec as codec  # noqa: E402
import eval_adaptive_lossless_test as evaluation  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_package(split: str, rules: dict) -> tuple[Path, Path]:
    source = (ROOT / "cold_start_streaming_20261014" / "pems07_full_cold_state.npz"
              if split == "val" else
              ROOT / "online_end_to_end_20261014" / "pems07_full_online_state.npz")
    sparse_source = CODEC_DIR / "packages" / "online" / f"pems07_s1_{split}_u16.npz"
    with np.load(source, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    raw = state["train_raw"]
    if not np.array_equal(raw, raw.astype(np.uint16)):
        raise ValueError("Full-node state cannot be losslessly cast to uint16")
    state["train_raw"] = raw.astype(np.uint16)
    with np.load(sparse_source, allow_pickle=False) as archive:
        sparse = {key: archive[key] for key in archive.files}
    nodes = sparse["node_ids"].astype(np.int64)
    if not np.array_equal(state["train_raw"][:, nodes], sparse["train_raw"]):
        raise AssertionError("Full and SRM history disagree")
    if not np.array_equal(state["quant_scale"][:, nodes], sparse["quant_scale"]):
        raise AssertionError("Full and SRM scale disagree")
    if len(nodes) != 121 or len(state["node_ids"]) != 883:
        raise AssertionError("Unexpected frozen node counts")
    target = HERE / f"pems07_full_exact_{split}.alsc"
    blob = evaluation.pack_fixed(state, rules)
    if target.exists():
        if target.read_bytes() != blob:
            raise FileExistsError(f"Existing package differs: {target}")
    else:
        target.write_bytes(blob)
    decoded = codec.decode(target.read_bytes())
    for key, expected in state.items():
        actual = decoded[key]
        if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise AssertionError(f"Round-trip failed: {split} {key}")
    return source, target


def main() -> None:
    val_exact = evaluation.VALIDATION / "pems07_val_exact.alsc"
    rules = evaluation.recipe(val_exact)
    sources, packages = {}, {}
    for split in ("val", "test"):
        source, package = build_package(split, rules)
        sources[split], packages[split] = source, package
        print(f"Built {split}: {package.stat().st_size:,} B", flush=True)
    result = evaluation.evaluate("PEMS07", packages["val"], packages["test"])
    result.update({
        "protocol": "PEMS07 frozen Student, full-node exact memory, same sparse-exact field recipes",
        "source_sha256": {key: sha256(path) for key, path in sources.items()},
        "package_sha256": {key: sha256(path) for key, path in packages.items()},
        "package_bytes": {key: path.stat().st_size for key, path in packages.items()},
        "test_based_selection": False,
        "calibration_fit_split": "validation_only",
    })
    output = HERE / "pems07_full_exact_metrics.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
