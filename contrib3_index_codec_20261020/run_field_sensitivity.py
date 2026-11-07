"""Where does each field's bit budget actually saturate?

The fd probe (`run_fd_sensitivity.py`) showed that fd tolerates 2 bits *and*
survives being zeroed, i.e. its precision-insensitivity is a property of the
confidence channel, not a compression achievement.  The same question must be
asked of every field before "minimum saturating configuration" can be claimed,
because a field that tolerates 0 bits can never support a claim about bit
budgets.

For each field this script walks a ladder of per-(h,node) symmetric integer
codes down to zero and records MAE.  The **knee** -- the coarsest code whose
cost still clears the 0.01 gate -- is the per-field answer.  Fields:

    T    the retrieved mean, the main payload (S,H,N)
    hd   mean history distance, a confidence feature (S,N)
    E    exact-match payload -- expected to be load-bearing (S,H,N)
    fd   dispersion, known insensitive (S,H,N)

Validation only, no training.  `zero` is the control: if a field is free to
delete, its knee is meaningless and must be reported as "unused" instead.

Usage:  python run_field_sensitivity.py PEMS04 PEMS08
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_law_config import EXP, HORIZON, evaluate, load_split  # noqa: E402
from emulate_and_build_residual_oracle import FrozenPipeline    # noqa: E402
import codec_fields as CF                                        # noqa: E402

OUT = HERE / "field_sensitivity.json"
GATE = 0.01


def pernode_quant(x, bits):
    """Symmetric per-(h,node) integer code, cast to the smallest dtype."""
    levels = (1 << (bits - 1)) - 1
    dtype = np.int16 if bits > 8 else np.int8
    x = np.asarray(x, dtype=np.float32)
    absmax = np.abs(x).max(axis=0)                       # (H,N) or (N,)
    scale = np.where(absmax > 0, absmax / float(levels), 1.0).astype(np.float32)
    q = np.clip(np.round(x / scale[None]), -levels, levels).astype(dtype)
    return (q.astype(np.float32) * scale[None]).astype(np.float32), q


def main():
    datasets = sys.argv[1:] or ["PEMS04", "PEMS08"]
    res = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    for ds in datasets:
        ds_dir = ROOT / "datasets" / ds
        exp_dir = ROOT / "EXP_experiments/TLAST_improved" / ds / EXP[ds]
        meta = np.load(ds_dir / f"{ds}_12to12_index_scaler.npz")
        idx_val = np.asarray(meta["idx_val"], dtype=np.int64)
        s_mean = float(np.asarray(meta["scaler_mean"]).item())
        s_std = float(np.asarray(meta["scaler_std"]).item())

        from utils.datasets_utils import file_path_dict, load_raw_data
        raw, _ = load_raw_data(file_path_dict[ds])
        raw = np.asarray(raw[..., 0], dtype=np.float32)
        n_nodes = raw.shape[1]

        bb = np.load(ds_dir / f"{ds}_12to12_backbone_predictions_joint3.npz")
        P = np.asarray(bb["backbone_pred_val"], dtype=np.float32)
        pipe = FrozenPipeline(exp_dir, HORIZON, n_nodes)
        pipe.set_codec(None, None, quantize=False)

        cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                        allow_pickle=False)
        val = load_split(cache, "val")
        y_raw = raw[idx_val[:, None] + np.arange(1, HORIZON + 1)[None], :]
        y = ((y_raw - s_mean) / s_std).astype(np.float32)
        v = y_raw != 0
        c0 = evaluate(pipe, P, y, v, s_std, val)

        per = {"C0": c0}
        print(f"=== {ds} val (C0 {c0:.5f}) ===", flush=True)
        print(f"{'field':6s} {'variant':8s} {'MAE':>9s} {'delta':>9s} "
              f"{'verdict':>8s}")
        for field in ("T", "hd", "E", "fd"):
            orig = val[field]
            rows = {}
            knee = None
            for bits in (16, 8, 6, 4, 2, 0):
                if bits == 0:
                    arr = np.zeros_like(orig)
                    name = "zero"
                elif field == "E" and bits == 16:
                    # E ships as a sparse fp16 payload, i.e. 16 bits, but not
                    # an integer code -- test the form actually deployed
                    arr = CF.decode_fd(CF.encode_fd(orig))
                    name = "f16"
                else:
                    arr, _ = pernode_quant(orig, bits)
                    name = f"int{bits}"
                alt = dict(val)
                alt[field] = arr
                mae = evaluate(pipe, P, y, v, s_std, alt)
                ok = (mae - c0) <= GATE
                rows[name] = dict(MAE=mae, delta=mae - c0, pass_gate=bool(ok))
                if ok and bits > 0:
                    knee = bits
                print(f"{field:6s} {name:8s} {mae:9.5f} {mae-c0:+9.5f} "
                      f"{'pass' if ok else 'FAIL':>8s}", flush=True)
                del alt, arr
            per[field] = dict(rows=rows, knee_bits=knee)
            print(f"       -> knee {knee} bits"
                  f"{'  (field unused: zero also passes)' if rows['zero']['pass_gate'] else ''}",
                  flush=True)
        res[ds] = per
        cache.close()
        del pipe, P, y, v, val
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
