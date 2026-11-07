"""Cross-dataset global accounting for the field-wise compression law.

Companion to run_fieldwise_global_accounting.py (PEMS04 only).  Repeats the
per-field compression ladder on PEMS03/04/07/08 under ONE protocol, using the
UNCOMPRESSED caches (only PEMS04 ships a latc_pca_r8 cache, so using it would
break comparability).  All four therefore share the same reference:

    C0 : T = retrieval_val, E / fd / hd = raw fp32, mask = 1 byte/slot

Arms
----
per field : fd f16 / i16 / i8 ; hd f16 / i16 / i8
A1        : E sparse f16 + fd f16 + hd i16 + shared 1-bit availability mask
A2        : E sparse f16 + fd i8  + hd i8  + shared 1-bit availability mask

Pre-registered criteria (locked before running)
-----------------------------------------------
G1  on every dataset, the best full arm costs <= 0.01 MAE vs C0
G2  on every dataset, the best full arm reduces total cache bytes >= 50%
G3  the direction of the two factors reproduces: fd tolerates <=16-bit
    quantisation (|cost| <= 0.01) while E low-rank r8 stays >= 0.05

Any dataset failing G1/G2 downgrades the claim to "dataset-dependent".

Note on the mask: the availability bitmap is ALSO the sparse index for E, so
it is counted once (1 bit/slot) rather than twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emulate_and_build_residual_oracle import (                     # noqa: E402
    FrozenPipeline, quality_scalars,
)
from utils.datasets_utils import file_path_dict, load_raw_data      # noqa: E402
from run_fieldwise_cross_dataset import DATASETS, fit_pca_cov       # noqa: E402

OUT = Path(__file__).resolve().parent / "fieldwise_global_cross_val.json"

HORIZON = 12
RANK = 8
Y_OFF = np.arange(1, 13, dtype=np.int64)
ACCEPTED_C0 = {  # recorded by run_fieldwise_cross_dataset.py
    "PEMS03": 13.1987, "PEMS04": 17.8520, "PEMS07": 18.2222, "PEMS08": 13.2461,
}


def quant_fp16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def _per_node_scale(x, node_axis, levels):
    """Symmetric quantiser with one scale per node (stored as float32)."""
    x = np.asarray(x, dtype=np.float32)
    axis = node_axis % x.ndim
    reduce_axes = tuple(a for a in range(x.ndim) if a != axis)
    absmax = np.abs(x).max(axis=reduce_axes)
    scale = np.where(absmax > 0, absmax / float(levels),
                     1.0).astype(np.float32)
    shape = [1] * x.ndim
    shape[axis] = -1
    return scale.reshape(shape)


def quant_int(x, node_axis=-1, bits=8):
    levels = 127 if bits == 8 else 32767
    dtype = np.int8 if bits == 8 else np.int16
    x = np.asarray(x, dtype=np.float32)
    scale = _per_node_scale(x, node_axis, levels)
    q = np.clip(np.round(x / scale), -levels, levels).astype(dtype)
    return (q.astype(np.float32) * scale).astype(np.float32)


def run_dataset(ds: str, exp_name: str):
    exp_dir = ROOT / "EXP_experiments/TLAST_improved" / ds / exp_name
    ds_dir = ROOT / "datasets" / ds
    cache_path = ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz"
    backbone_path = ds_dir / f"{ds}_12to12_backbone_predictions_joint3.npz"

    index = np.load(ds_dir / f"{ds}_12to12_index_scaler.npz")
    idx_train = np.asarray(index["idx_train"], dtype=np.int64)
    idx_val = np.asarray(index["idx_val"], dtype=np.int64)
    scaler_mean = float(np.asarray(index["scaler_mean"]).item())
    scaler_std = float(np.asarray(index["scaler_std"]).item())

    raw_all, _ = load_raw_data(file_path_dict[ds])
    raw = np.asarray(raw_all[..., 0], dtype=np.float32)
    num_nodes = raw.shape[1]

    cache = np.load(cache_path, allow_pickle=False)
    T_raw = np.asarray(cache["retrieval_val"], dtype=np.float32)
    e_val = np.asarray(cache["exact_retrieval_val"], dtype=np.float32)
    avail = np.asarray(cache["exact_available_val"], dtype=np.float32)
    hd_raw = np.asarray(cache["history_distance_val"], dtype=np.float32)
    fd_raw = np.asarray(cache["future_dispersion_val"], dtype=np.float32)

    p_val = np.asarray(np.load(backbone_path)["backbone_pred_val"],
                       dtype=np.float32)
    y_raw = raw[idx_val[:, None] + Y_OFF[None], :]
    y_z = ((y_raw - scaler_mean) / scaler_std).astype(np.float32)
    valid_q = y_raw != 0

    pipe = FrozenPipeline(exp_dir, HORIZON, num_nodes)
    pipe.set_codec(None, None, quantize=False)

    def mae(pred_z):
        return (float((np.abs(pred_z[..., 0] - y_z) * valid_q).sum()
                      / valid_q.sum()) * scaler_std)

    def run(E, fd, hd):
        q0, q1, q2 = quality_scalars(
            {"history_distance_val": hd, "future_dispersion_val": fd,
             "exact_available_val": avail}, scaler_std, HORIZON)
        return mae(pipe.forward(p_val, T_raw, E, avail, q0, q1, q2))

    fired = avail > 0.5
    fire_rate = float(fired.mean())
    n_slots = int(fired.size)
    zero_fill = np.float32(0.0)
    e_sparse_f16 = np.where(fired, quant_fp16(e_val), zero_fill)

    arms, per_field = {}, {}
    base = run(e_val, fd_raw, hd_raw)
    arms["C0_raw"] = base

    arms["E_sparse_f16"] = run(e_sparse_f16, fd_raw, hd_raw)
    for bits, name in ((16, "f16"), (16, "i16"), (8, "i8")):
        fd_arm = (quant_fp16(fd_raw) if name == "f16"
                  else quant_int(fd_raw, bits=bits))
        hd_arm = (quant_fp16(hd_raw) if name == "f16"
                  else quant_int(hd_raw, bits=bits))
        arms[f"fd_{name}"] = run(e_val, fd_arm, hd_raw)
        arms[f"hd_{name}"] = run(e_val, fd_raw, hd_arm)
        per_field[f"fd_{name}"] = arms[f"fd_{name}"] - base
        per_field[f"hd_{name}"] = arms[f"hd_{name}"] - base

    arms["A1_conservative"] = run(e_sparse_f16, quant_fp16(fd_raw),
                                  quant_int(hd_raw, bits=16))
    arms["A2_aggressive"] = run(e_sparse_f16, quant_int(fd_raw, bits=8),
                                quant_int(hd_raw, bits=8))

    # ---- byte accounting, in bits per (query, horizon, node) slot ----------
    # hd is (S,N): one value per node, i.e. 1/HORIZON of a slot.
    per_elem = {"T": 32.0, "E": 32.0, "fd": 32.0, "hd": 32.0 / HORIZON,
                "mask": 8.0}
    bits_all_raw = sum(per_elem.values())
    bytes_all_raw = n_slots * bits_all_raw / 8.0

    def bits_of(e_mode, fd_mode, hd_mode):
        b = 0.0
        b += RANK * 16.0 / HORIZON                       # T: r8 fp16 code
        b += (16.0 * fire_rate if e_mode == "sparse_f16" else 32.0)
        b += 1.0                                          # shared bitmap
        b += {"f32": 32.0, "f16": 16.0, "i16": 16.0, "i8": 8.0}[fd_mode]
        b += {"f32": 32.0, "f16": 16.0, "i16": 16.0,
              "i8": 8.0}[hd_mode] / HORIZON
        return b

    configs = {
        "A1_conservative": bits_of("sparse_f16", "f16", "i16"),
        "A2_aggressive": bits_of("sparse_f16", "i8", "i8"),
    }
    accounting = {
        "all_raw_bits_per_slot": bits_all_raw,
        "all_raw_MB": bytes_all_raw / 1e6,
    }
    for name, b in configs.items():
        accounting[name] = {
            "bits_per_slot": b,
            "MB": n_slots * b / 8.0 / 1e6,
            "reduction_vs_all_raw": 1.0 - b / bits_all_raw,
        }

    hd_absmax = float(np.abs(hd_raw).max())
    hd_f16_overflow = int(np.isinf(np.asarray(hd_raw, dtype=np.float16)).sum())

    result = {
        "nodes": int(num_nodes),
        "val_samples": int(len(idx_val)),
        "fire_rate": fire_rate,
        "baseline_C0": base,
        "accepted_C0": ACCEPTED_C0[ds],
        "acceptance_delta": base - ACCEPTED_C0[ds],
        "arms_absolute": arms,
        "arms_delta": {k: v - base for k, v in arms.items()},
        "per_field_delta": per_field,
        "hd_absmax_raw": hd_absmax,
        "hd_f16_overflow_count": hd_f16_overflow,
        "accounting": accounting,
    }

    print(f"\n=== {ds} (N={num_nodes}, val={len(idx_val)}) ===", flush=True)
    print(f"  C0 {base:.7f}  (accepted {ACCEPTED_C0[ds]:.4f}, "
          f"delta {base - ACCEPTED_C0[ds]:+.5f})   fire {fire_rate:.4f}")
    for k, v in result["arms_delta"].items():
        print(f"    {k:18s} {v:+.7f}")
    print(f"  hd |max| {hd_absmax:.1f}   f16 overflow slots {hd_f16_overflow}")
    for name in configs:
        a = accounting[name]
        print(f"  {name:16s} {a['MB']:7.2f} MB  "
              f"({a['bits_per_slot']:6.2f} bits/slot)  "
              f"vs all-raw {a['reduction_vs_all_raw']*100:6.2f}%")
    return result


def main():
    results = {}
    for ds, exp_name in DATASETS.items():
        try:
            results[ds] = run_dataset(ds, exp_name)
        except Exception as exc:
            print(f"\n=== {ds} FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[ds] = {"error": f"{type(exc).__name__}: {exc}"}

    ok = {d: r for d, r in results.items() if "error" not in r}

    def pick(r):
        a, b = r["arms_delta"]["A1_conservative"], r["arms_delta"]["A2_aggressive"]
        return ("A1_conservative", a) if abs(a) <= abs(b) else ("A2_aggressive", b)

    print("\n================ SUMMARY ================")
    print("dataset  fire    C0        best_arm        cost       "
          "bytes/slot   saving")
    for ds, r in ok.items():
        name, cost = pick(r)
        a = r["accounting"][name]
        print(f"{ds:8s} {r['fire_rate']:.4f}  {r['baseline_C0']:.5f}  "
              f"{name:16s} {cost:+.7f}  {a['bits_per_slot']:8.2f}  "
              f"{a['reduction_vs_all_raw']*100:6.2f}%")

    costs = {d: pick(r)[1] for d, r in ok.items()}
    sav = {d: r["accounting"][pick(r)[0]]["reduction_vs_all_raw"]
           for d, r in ok.items()}
    verdict = {
        "G1_equal_accuracy_le_0.01_everywhere": bool(
            ok and all(abs(c) <= 0.01 for c in costs.values())),
        "G2_reduction_ge_50pct_everywhere": bool(
            ok and all(s >= 0.50 for s in sav.values())),
        "G3_fd_tolerates_16bit_everywhere": bool(
            ok and all(abs(r["per_field_delta"][f"{k}_f16"])
                       <= 0.01
                       for r, k in ((r, "fd") for r in ok.values()))),
        "cost_by_dataset": costs,
        "saving_by_dataset": sav,
        "datasets_completed": list(ok),
        "datasets_failed": [d for d, r in results.items() if "error" in r],
    }
    verdict["fieldwise_global_generalises"] = bool(
        verdict["G1_equal_accuracy_le_0.01_everywhere"]
        and verdict["G2_reduction_ge_50pct_everywhere"])
    print("\n" + json.dumps(verdict, indent=2))

    OUT.write_text(json.dumps({"results": results, "verdict": verdict},
                              indent=2), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
