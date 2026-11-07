"""Freeze PEMS-BAY SRM nodes and build exact float32 retrieval-state packages.

Only Train observations and Validation-derived node scores are read. Test
predictions and labels are deliberately not opened here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "contrib3_index_codec_20261020")]

from contrib3_index_codec_20261020.adaptive_lossless_state_codec import decode, encode
from utils.online_residual_memory import build_cold_start_state, build_train_only_state, packed_state

CACHE = ROOT / "datasets/PEMSBAY/PEMSBAY_12to12_analog_k8_exact_mean_712.npz"
INDEX = ROOT / "datasets/PEMSBAY/PEMSBAY_12to12_index_scaler_712.npz"
SCREEN = ROOT / "paper_results/pemsbay712_srm_validation_pilot_20261027/validation_screen.json"
CKPT = ROOT / "EXP_experiments/TLAST_improved/PEMSBAY/TLAST_improved_PEMSBAY_12to12_exppemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1/best.pth"
OUT = HERE / "freeze_manifest.json"


def digest(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hash_obj.update(block)
    return hash_obj.hexdigest()


def write_blob(path: Path, blob: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(blob)


def package(name: str, state: dict[str, np.ndarray]) -> dict[str, int]:
    npz = packed_state(state)
    optimized, direct, _ = encode(state)
    decoded = decode(optimized)
    for field, value in state.items():
        restored = decoded[field]
        if restored.dtype != value.dtype or restored.shape != value.shape or restored.tobytes() != value.tobytes():
            raise AssertionError(f"{name}: non-exact round trip for {field}")
    write_blob(HERE / f"{name}.npz", npz)
    write_blob(HERE / f"{name}.alsc", optimized)
    write_blob(HERE / f"{name}_direct.alsc", direct)
    return {"npz_bytes": len(npz), "alsc_bytes": len(optimized), "direct_alsc_bytes": len(direct)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if screen["test_evaluated"] or screen["protocol"] != "first55_validation_node_gain_screen_v1":
        raise ValueError("Unexpected Validation screen")
    selected = next(item for item in screen["budgets"] if item["fraction"] == 0.8)
    ids = np.asarray(selected["selected_nodes"], dtype=np.int32)
    if len(ids) != 260 or not np.all(np.diff(ids) > 0):
        raise ValueError("Expected 260 unique sorted nodes")
    with np.load(INDEX, allow_pickle=False) as archive:
        first_val_endpoint = int(archive["idx_val"][0])
    print("Building selected Train-only state", flush=True)
    sparse = build_train_only_state("PEMSBAY", CACHE, ids)
    print("Building full Train-only state", flush=True)
    full = build_train_only_state("PEMSBAY", CACHE, np.arange(325, dtype=np.int32))
    if sparse["train_raw"].dtype != np.float32 or np.array_equal(sparse["train_raw"], np.rint(sparse["train_raw"])):
        raise ValueError("PEMS-BAY float32 history not found")
    print("Packaging selected state", flush=True)
    sparse_sizes = package("selected_train_state", sparse)
    print("Packaging full state", flush=True)
    full_sizes = package("full_train_state", full)
    print("Packaging causal Validation cold-start state", flush=True)
    sparse_cold = build_cold_start_state(sparse, first_val_endpoint)
    cold_sizes = package("selected_val_cold_state", sparse_cold)
    report = {
        "protocol": "pemsbay_float32_exact_srm_package_v1",
        "dataset": "PEMSBAY",
        "test_evaluated": False,
        "selection": "Validation first55 gain ranking; 80pct budget chosen by later Validation raw MAE",
        "selected_nodes": ids.tolist(),
        "total_nodes": 325,
        "checkpoint_sha256": digest(CKPT),
        "validation_screen_sha256": digest(SCREEN),
        "cache_sha256": digest(CACHE),
        "first_val_endpoint": first_val_endpoint,
        "history_dtype": "float32_exact",
        "quantized_nodes": 0,
        "codec": "ALSC lossless float32; no numeric delta, per-field reversible layout/compression",
        "sizes": {"selected_train": sparse_sizes, "full_train": full_sizes,
                  "selected_val_cold": cold_sizes},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(OUT), "sizes": report["sizes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
