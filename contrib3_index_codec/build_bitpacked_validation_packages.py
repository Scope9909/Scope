"""Persist and verify self-describing bitpacked s1 Validation state packages."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"
OUT_DIR = ONLINE / "bitpacked_val"
MANIFEST = ONLINE / "bitpacked_val_package_manifest.json"
DATASETS = ("pems03", "pems04", "pems07", "pems08")
sys.path.insert(0, str(HERE))
import pilot_reversible_predictive_coding as base_codec  # noqa: E402
import delta_bitpacked_state_codec as codec  # noqa: E402
from online_codec_runtime import decode_online_state  # noqa: E402


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        source = ONLINE / f"{dataset}_s1_val_u16.npz"
        target = OUT_DIR / f"{dataset}_s1_val_delta_bitpacked.npz"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing package: {target}")
        info = codec.save(source, target)
        decoded = decode_online_state(target)
        with np.load(source, allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        if decoded.keys() != original.keys():
            raise AssertionError(f"{dataset}: decoded field set changed")
        for key, value in original.items():
            expected = (np.asarray(value, dtype=np.float32)
                        if key == "train_raw" else value)
            if not np.array_equal(decoded[key], expected):
                raise AssertionError(f"{dataset}: decoded field mismatch: {key}")
        baseline_state, _ = base_codec.normalize_state(source)
        baseline_blob, _ = base_codec.best_bytes(baseline_state)
        rows.append({
            "dataset": dataset.upper(),
            "source": str(source),
            "package": str(target),
            "source_npz_bytes": source.stat().st_size,
            "strong_lossless_same_layout_bytes": len(baseline_blob),
            "bitpacked_package_bytes": target.stat().st_size,
            "saved_vs_strong_lossless_bytes": len(baseline_blob) - target.stat().st_size,
            "saved_fraction": 1 - target.stat().st_size / len(baseline_blob),
            "codec_version": 1,
            "array_order": info["array_order"],
            "exact_all_fields_roundtrip": True,
            "sha256": info["checksum"],
        })
    MANIFEST.write_text(json.dumps({
        "title": "Self-describing exact delta-bitpacked Validation packages",
        "date": "2026-10-23",
        "scope": "s1 Validation online states only; Test not used",
        "decoder": "contrib3_index_codec/online_codec_runtime.py",
        "results": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"\nWrote manifest {MANIFEST}")


if __name__ == "__main__":
    main()
