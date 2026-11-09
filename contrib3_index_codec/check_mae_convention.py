"""Close the test-split gap: is it a metric convention, or a model difference?

STEP 0 reproduced validation bit-for-bit but reported three test splits as
MISMATCH, always in the direction "real path is *better* than the log".  A
plausible cause is that the training log reports the mean of 12 per-horizon
MAEs while this harness reports one global element-wise mean; the two differ
whenever the valid-element count is not identical across horizons.

This script recomputes both conventions from the *same* dumped errors, so the
question becomes an equation instead of an argument.  If the per-horizon
convention lands on the logged number, the gap is fully explained and the real
path is confirmed; if it does not, the gap is still open and must not be waved
through.

Usage:  python check_mae_convention.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Test MAE as printed by each dataset's own training log.
LOG = {
    "PEMS03": 14.1924,
    "PEMS04": 17.6296,
    "PEMS07": 18.8438,
    "PEMS08": 12.9798,
}


def conventions(abs_err, valid, horizon_axis):
    """Every averaging convention the training log might have used.

    ``global``      one mean over all valid elements  (what the harness reports)
    ``per_horizon`` mean of the 12 per-horizon means
    ``per_slot``    mean of the 12*N per-(horizon,node) means -- this one
                    re-weights every slot equally instead of by sample count,
                    so it differs from ``global`` whenever valid counts vary
                    across slots, which they do (measured 3121..3567).
    """
    m = np.where(valid, abs_err, np.nan)
    v = np.where(valid, 1.0, np.nan)
    global_mae = float(np.nanmean(m))

    per_h = np.moveaxis(m, horizon_axis, 0).reshape(
        abs_err.shape[horizon_axis], -1)
    with np.errstate(invalid="ignore"):
        h_mae = np.array([np.nanmean(per_h[h]) for h in range(per_h.shape[0])])

    # (H, N, Q) -> per-(horizon,node) means, then one mean over all slots
    perm = [horizon_axis] + [a for a in range(m.ndim) if a != horizon_axis]
    mm = np.transpose(m, perm)
    vv = np.transpose(v, perm)
    with np.errstate(invalid="ignore"):
        slot_sum = np.nansum(mm, axis=tuple(range(1, mm.ndim)))
        slot_cnt = np.nansum(vv, axis=tuple(range(1, mm.ndim)))
    slot_mae = slot_sum / np.where(slot_cnt > 0, slot_cnt, np.nan)
    return (global_mae, float(np.nanmean(h_mae)), h_mae,
            float(np.nanmean(slot_mae)))


def main():
    TOL = 5e-3
    print(f"{'ds':8s} {'global':>10s} {'per-horiz':>10s} {'per-slot':>10s} "
          f"{'log':>10s}  best-match")
    verdicts = {}
    for ds, logged in LOG.items():
        f = HERE / f"_sloterr_{ds}_test.npz"
        if not f.exists():
            print(f"{ds:8s}  missing {f.name}")
            continue
        z = np.load(f)
        ae = np.asarray(z["absolute_error"], dtype=np.float64)
        va = np.asarray(z["valid"], dtype=bool)
        j = json.loads((HERE / f"_align_{ds}_test.json").read_text("utf-8"))
        reported = float(j["metrics"]["mae"])

        axis = next((a for a in range(ae.ndim) if ae.shape[a] == 12), None)
        if axis is None:
            print(f"{ds:8s}  no axis of length 12 in shape {ae.shape}")
            continue
        g, p, h_mae, s = conventions(ae, va, axis)

        cands = {"global": g, "per_horizon": p, "per_slot": s}
        best_name = min(cands, key=lambda k: abs(cands[k] - logged))
        best_val = cands[best_name]
        ok = abs(best_val - logged) < TOL
        verdicts[ds] = dict(shape=list(ae.shape), horizon_axis=int(axis),
                            conventions=cands, logged=logged,
                            harness_reported=reported,
                            best_match=best_name,
                            gap_vs_log=best_val - logged, explained=bool(ok))
        print(f"{ds:8s} {g:10.5f} {p:10.5f} {s:10.5f} {logged:10.4f}  "
              f"{best_name} {'EXPLAINED' if ok else 'STILL OPEN'} "
              f"({best_val - logged:+.5f})")

    (HERE / "mae_convention_check.json").write_text(
        json.dumps(verdicts, indent=2), encoding="utf-8")
    n = sum(1 for v in verdicts.values() if v["explained"])
    print(f"\nexplained {n}/{len(verdicts)}")
    return 0 if n == len(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
