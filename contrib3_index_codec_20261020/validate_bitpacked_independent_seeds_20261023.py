"""Independent checkpoint confirmation for exact delta-bitpacked states."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "packages" / "online" / "bitpacked_independent_seed_confirmation_20261023.json"

spec = importlib.util.spec_from_file_location(
    "bitpacked_fullval", HERE / "validate_bitpacked_fullval_20261023.py")
fullval = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fullval)


def main() -> None:
    # PEMS03/PEMS08 s2/s3 reuse their s1 checkpoint; only independent models run.
    targets = (("PEMS04", 2), ("PEMS04", 3), ("PEMS07", 2), ("PEMS07", 3))
    results = [fullval.run_dataset(dataset, seed) for dataset, seed in targets]
    result = {
        "title": "Delta-bitpacked full Validation independent-checkpoint confirmation",
        "date": "2026-10-23",
        "scope": "Only distinct PEMS04/PEMS07 s2/s3 checkpoints; Test not loaded/evaluated.",
        "results": results,
        "limitations": ["Single full Validation pass per checkpoint", "Runtime seconds are single-run diagnostics"],
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
