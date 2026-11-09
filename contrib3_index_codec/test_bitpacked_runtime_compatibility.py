"""Compatibility and corruption checks for the extended online state decoder."""
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"
PACKAGES = ONLINE / "bitpacked_val"
OUT = ONLINE / "bitpacked_runtime_compatibility.json"


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        "online_codec_runtime", HERE / "online_codec_runtime.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runtime = load_runtime()
    checks = []
    for dataset in ("pems03", "pems04", "pems07", "pems08"):
        candidate = PACKAGES / f"{dataset}_s1_val_delta_bitpacked.npz"
        decoded = runtime.decode_online_state(candidate)
        with np.load(ONLINE / f"{dataset}_s1_val_u16.npz", allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        for key, value in original.items():
            expected = np.asarray(value, dtype=np.float32) if key == "train_raw" else value
            if not np.array_equal(decoded[key], expected):
                raise AssertionError(f"{dataset}: new decoder differs at {key}")
        checks.append({"format": "new_delta_bitpacked", "dataset": dataset.upper(),
                       "fields_exact": True})
    for name in ("pems08_s1_val_u16.npz", "pems08_s1_val_fixed_19_idx.npz"):
        decoded = runtime.decode_online_state(ONLINE / name)
        if "train_raw" not in decoded or "idx_train" not in decoded:
            raise AssertionError(f"Legacy archive did not decode: {name}")
        checks.append({"format": "legacy", "archive": name,
                       "decoded": True})

    source = PACKAGES / "pems08_s1_val_delta_bitpacked.npz"
    with np.load(source, allow_pickle=False) as archive:
        corrupted = {key: archive[key] for key in archive.files}
    checksum = corrupted["train_raw_delta_sha256"].copy()
    checksum[0] ^= 1
    corrupted["train_raw_delta_sha256"] = checksum
    rejected = False
    with tempfile.TemporaryDirectory(prefix="bitpack_corrupt_", dir=ONLINE) as td:
        bad_path = Path(td) / "corrupted.npz"
        np.savez_compressed(bad_path, **corrupted)
        try:
            runtime.decode_online_state(bad_path)
        except ValueError as error:
            rejected = "checksum mismatch" in str(error)
    if not rejected:
        raise AssertionError("Corrupted checksum was not rejected")
    result = {"title": "Bitpacked runtime compatibility checks",
              "date": "2026-10-23", "checks": checks,
              "corrupt_checksum_rejected": rejected}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
