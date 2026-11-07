"""Read-only estimate of a lossless dtype encoding for sparse online memory."""

import io
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "PEMS03": "pems03_int8_uniform_20261014/sparse_online_state.npz",
    "PEMS04": "paper_results/icde_retrieval_free_pilot_20261012/online_state_budget_pems04_s1_cap2905847_20261013/online_train_state.npz",
    "PEMS07": "paper_results/icde_retrieval_free_pilot_20261012/pems07_online_state_audit_s1_20261013/robust_train_state.npz",
    "PEMS08": "paper_results/icde_retrieval_free_pilot_20261012/online_state_budget_pems08_s1_cap2785289_20261013/online_train_state.npz",
}

for dataset, relative in PATHS.items():
    path = ROOT / relative
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    raw = state["train_raw"]
    if not (np.isfinite(raw).all() and np.array_equal(raw, np.rint(raw))
            and 0 <= raw.min() and raw.max() <= 65535):
        raise ValueError(f"{dataset}: train_raw cannot be losslessly uint16 encoded")
    state["train_raw"] = raw.astype(np.uint16)
    if not np.array_equal(state["train_raw"].astype(np.float32), raw):
        raise ValueError(f"{dataset}: roundtrip mismatch")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **state)
    original = path.stat().st_size
    encoded = buffer.tell()
    print(dataset, original, encoded, round(100 * (1 - encoded / original), 2))
