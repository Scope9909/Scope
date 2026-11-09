"""Post-hoc paired Test audit; no model or threshold selection.

Computes moving-block bootstrap CIs for the already frozen online selections,
and separately an offline equal-call-count ranking diagnostic.
"""

import json
import argparse
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
def mae(student, teacher, counts, selected, indices):
    chosen = selected[indices]
    errors = np.where(chosen[:, None], teacher[indices], student[indices])
    return float(np.mean(errors.sum(axis=0) / counts[indices].sum(axis=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", action="store_true", help="Audit the formal full-Validation calibrated run")
    args = parser.parse_args()
    suffix = "_formal" if args.formal else ""
    source = HERE / f"pems07{suffix}_calibrated_gate_final_test_paired_audit.npz"
    with np.load(source, allow_pickle=False) as z:
        endpoints = z["endpoints"].astype(np.int64)
        student = z["student_error_by_horizon"].astype(np.float64)
        teacher = z["teacher_error_by_horizon"].astype(np.float64)
        counts = z["valid_count_by_horizon"].astype(np.float64)
        scores = {
            "gate": z["gate_score"].astype(np.float64),
            "volatility": z["volatility_score"].astype(np.float64),
            "student_mean": z["student_mean_score"].astype(np.float64),
        }
        selected = {
            "gate": z["gate_selected"].astype(bool),
            "volatility": z["volatility_selected"].astype(bool),
            "student_mean": z["student_mean_selected"].astype(bool),
        }
    n = len(endpoints)
    if not np.all(np.diff(endpoints) == 1):
        raise ValueError("Test endpoints are not consecutive")
    full = np.arange(n)
    online_mae = {name: mae(student, teacher, counts, flag, full)
                  for name, flag in selected.items()}
    # Each score ranks the same number of Test windows as the frozen gate.
    # Ranking by the full Test score distribution is not online-deployable.
    k = int(selected["gate"].sum())
    equal_call = {}
    for name, score in scores.items():
        flag = np.zeros(n, dtype=bool)
        flag[np.argsort(score, kind="mergesort")[-k:]] = True
        equal_call[name] = {
            "teacher_windows": k,
            "mae": mae(student, teacher, counts, flag, full),
        }
    blocks = {}
    for i, index in enumerate(np.array_split(full, 4), start=1):
        base = mae(student, teacher, counts, np.zeros(n, dtype=bool), index)
        blocks[str(i)] = {
            "windows": len(index),
            "gate_gain": base-mae(student, teacher, counts, selected["gate"], index),
            "gate_vs_student_mean": (
                mae(student, teacher, counts, selected["student_mean"], index)
                -mae(student, teacher, counts, selected["gate"], index)
            ),
        }
    # Overlapping moving-block bootstrap preserves local serial dependence.
    # A 288-endpoint block is one day at 5-minute resolution.
    rng = np.random.default_rng(20261018)
    block_length = 288
    repeats = 3000
    stats = np.empty((repeats, 3), dtype=np.float64)
    starts = np.arange(n-block_length+1)
    chunks = int(np.ceil(n/block_length))
    for r in range(repeats):
        chosen_starts = rng.choice(starts, size=chunks, replace=True)
        index = np.concatenate([np.arange(s, s+block_length) for s in chosen_starts])[:n]
        base = mae(student, teacher, counts, np.zeros(n, dtype=bool), index)
        gate = mae(student, teacher, counts, selected["gate"], index)
        volatility = mae(student, teacher, counts, selected["volatility"], index)
        student_mean = mae(student, teacher, counts, selected["student_mean"], index)
        stats[r] = (base-gate, volatility-gate, student_mean-gate)
    labels = ("gain_vs_student", "advantage_vs_volatility", "advantage_vs_student_mean")
    ci = {name: {
        "point": float(value),
        "ci95": [float(np.quantile(stats[:, j], 0.025)),
                 float(np.quantile(stats[:, j], 0.975))],
    } for j, (name, value) in enumerate(zip(labels, (
        mae(student, teacher, counts, np.zeros(n, dtype=bool), full)-online_mae["gate"],
        online_mae["volatility"]-online_mae["gate"],
        online_mae["student_mean"]-online_mae["gate"],
    )))}
    report = {
        "protocol": "posthoc_PEMS07_Test_paired_moving_block_bootstrap_no_retuning",
        "n_windows": n,
        "online_teacher_calls": {name: int(v.sum()) for name, v in selected.items()},
        "online_mae": online_mae,
        "offline_equal_call_count": equal_call,
        "temporal_quarters": blocks,
        "bootstrap": {
            "unit": "overlapping contiguous 288-endpoint blocks (one day)",
            "block_length": block_length,
            "repeats": repeats,
            "positive_advantage_means_gate_better": True,
            "results": ci,
        },
        "caveat": "Test has previously been inspected in this project. Equal-call top-k is an offline diagnostic, not a deployable policy. Intervals condition on fixed checkpoints and gate; they do not measure training-seed uncertainty.",
    }
    out = HERE / f"pems07{suffix}_final_test_route_robustness.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
