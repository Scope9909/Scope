"""Prove that PEMS04 old three-seed Test states equal final ALSC states."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import numpy as np
from adaptive_lossless_state_codec import load_runtime_state
PAIRINGS = (
    ("Validation", HERE / "packages/online/adaptive_lossless_val_20261023_v2/pems04_val_selected.alsc",
     ROOT / "cold_start_streaming_20261014/pems04_sparse_cold_state.npz"),
    ("Test", HERE / "packages/online/adaptive_lossless_test_20261023_v1/pems04_test_selected.alsc",
     ROOT / "paper_results/icde_retrieval_free_pilot_20261012/online_state_budget_pems04_s1_cap2905847_20261013/online_train_state.npz"),
)
OUTPUT = HERE / "packages/online/pems04_three_seed_state_equivalence_20261024.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = []
    for split, new_path, old_path in PAIRINGS:
        modern = load_runtime_state(new_path)
        with np.load(old_path, allow_pickle=False) as archive:
            previous = {key: archive[key] for key in archive.files}
        keys_equal = set(modern) == set(previous)
        equal = keys_equal and all(np.array_equal(modern[key], previous[key])
                                   for key in modern)
        rows.append(dict(split=split, final_state=str(new_path),
                         previous_frozen_state=str(old_path),
                         fields=sorted(modern), field_count=len(modern),
                         identical_keys=keys_equal, all_fields_equal=equal))
        if not equal:
            raise AssertionError(f"PEMS04 {split} state changed")
    OUTPUT.write_text(json.dumps(dict(dataset="PEMS04", rows=rows,
                                      prior_three_seed_test_reusable=True),
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print("PEMS04 Validation and Test states are identical across old/new formats")


if __name__ == "__main__":
    main()
