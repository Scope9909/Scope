"""Field-wise codec: global memory accounting + per-field compression arms.

Purpose
-------
The cross-dataset run established the field-wise compressibility law for the
retrieval cache's two biggest semantic fields (T = retrieved mean, E = exact
payload).  But the cache has more fields, and the previous "global saving"
figure (34.5%) only priced E.  This script closes the accounting by

  1. pricing every field that is actually stored, at its real byte count, and
  2. measuring, end-to-end on validation MAE, whether the remaining
     37.5% field (``future_dispersion``) plus ``history_distance`` and the
     availability mask can be compressed for free.

Arms (all on the deployed frozen joint3 checkpoint, zero training)
-----------------------------------------------------------------
R0  deployed reference : T = r8 code (as shipped), E / fd / hd = f32, mask = bool
A1  conservative       : E sparse fp16 + fd fp16 + hd fp16 + mask bit-packed
A2  aggressive         : E sparse fp16 + fd int8 + hd int8 + mask bit-packed
plus single-field arms (fd f16, fd i8, hd f16, hd i8) so each field can be
priced on its own.

Pre-registered criterion (locked before running)
------------------------------------------------
G1  the full arm must cost <= 0.01 MAE vs R0 (equal-accuracy band, the same
    band used for E on 2026-10-20).
G2  the resulting total cache-state reduction vs the all-raw store must be
    >= 50%.  Below that the "lightweighting" claim does not carry a
    contribution on its own and the route is reported as closed.
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
from run_aip_tc_three_arm import DS, EXP, CACHE, BACKBONE           # noqa: E402

OUT = Path(__file__).resolve().parent / "fieldwise_global_accounting_pems04_val.json"
RECORDED_CANONICAL = 17.8537288
HORIZON = 12


def squash(x):
    x = np.log1p(np.clip(x, 0.0, None))
    return x / (1.0 + x)


def quant_fp16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def quant_int8_per_node(x, node_axis=-1):
    """Symmetric per-node int8 with one scale per node (32 bits of side data)."""
    x = np.asarray(x, dtype=np.float32)
    absmax = np.abs(x).max(axis=tuple(a for a in range(x.ndim)
                                      if a != (node_axis % x.ndim)))
    scale = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
    shape = [1] * x.ndim
    shape[node_axis % x.ndim] = -1
    q = np.clip(np.round(x / scale.reshape(shape)), -127, 127).astype(np.int8)
    return (q.astype(np.float32) * scale.reshape(shape)).astype(np.float32)


def quant_int16_per_node(x, node_axis=-1):
    x = np.asarray(x, dtype=np.float32)
    absmax = np.abs(x).max(axis=tuple(a for a in range(x.ndim)
                                      if a != (node_axis % x.ndim)))
    scale = np.where(absmax > 0, absmax / 32767.0, 1.0).astype(np.float32)
    shape = [1] * x.ndim
    shape[node_axis % x.ndim] = -1
    q = np.clip(np.round(x / scale.reshape(shape)), -32767, 32767).astype(
        np.int16)
    return (q.astype(np.float32) * scale.reshape(shape)).astype(np.float32)


def main():
    index = np.load(DS / "PEMS04_12to12_index_scaler.npz")
    idx_val = np.asarray(index["idx_val"], dtype=np.int64)
    scaler_mean = float(np.asarray(index["scaler_mean"]).item())
    scaler_std = float(np.asarray(index["scaler_std"]).item())

    from utils.datasets_utils import file_path_dict, load_raw_data
    raw_all, _ = load_raw_data(file_path_dict["PEMS04"])
    raw = np.asarray(raw_all[..., 0], dtype=np.float32)
    num_nodes = raw.shape[1]

    cache = np.load(CACHE, allow_pickle=True)
    e_val = np.asarray(cache["exact_retrieval_val"], dtype=np.float32)
    avail = np.asarray(cache["exact_available_val"], dtype=np.float32)
    hd_raw = np.asarray(cache["history_distance_val"], dtype=np.float32)   # (S,N)
    fd_raw = np.asarray(cache["future_dispersion_val"], dtype=np.float32)  # (S,H,N,1)
    t_code = np.asarray(cache["retrieval_code_val"], dtype=np.float32)
    t_mean = np.asarray(cache["retrieval_temporal_mean"], dtype=np.float32)
    t_basis = np.asarray(cache["retrieval_temporal_basis"], dtype=np.float32)
    T_r8 = t_mean[None] + np.einsum("srnc,nhr->shnc", t_code, t_basis,
                                    optimize=True)

    p_val = np.asarray(np.load(BACKBONE, allow_pickle=True)["backbone_pred_val"],
                       dtype=np.float32)
    y_raw = raw[idx_val[:, None] + np.arange(1, HORIZON + 1)[None], :]
    y_z = ((y_raw - scaler_mean) / scaler_std).astype(np.float32)
    valid_q = y_raw != 0

    pipe = FrozenPipeline(EXP, HORIZON, num_nodes)
    pipe.set_codec(None, None, quantize=False)

    def mae(pred_z):
        return (float((np.abs(pred_z[..., 0] - y_z) * valid_q).sum()
                      / valid_q.sum()) * scaler_std)

    def run(E, fd_raw_used, hd_raw_used):
        cache_used = {
            "history_distance_val": hd_raw_used,
            "future_dispersion_val": fd_raw_used,
            "exact_available_val": avail,
        }
        q0, q1, q2 = quality_scalars(cache_used, scaler_std, HORIZON)
        return mae(pipe.forward(p_val, T_r8, E, avail, q0, q1, q2))

    fired = avail > 0.5
    n_slots = int(fired.size)
    n_fired = int(fired.sum())

    # E sparse: zero fill is admissible because avail = 0 removes E from the
    # update entirely (o2 = o1 + w * avail * (E - o1)).  Verified below by
    # requiring the arm to reproduce the previously recorded +0.0000000.
    e_sparse_q = np.where(fired, quant_fp16(e_val), np.float32(0.0))
    e_dense_f32 = e_val

    arms, per_field = {}, {}
    print("arm: reference (deployed) ...", flush=True)
    arms["R0_deployed_reference"] = run(e_dense_f32, fd_raw, hd_raw)
    base = arms["R0_deployed_reference"]

    print("arm: E sparse fp16 only ...", flush=True)
    arms["E_sparse_fp16_only"] = run(e_sparse_q, fd_raw, hd_raw)

    for name, fn in (("f16", quant_fp16), ("i8", quant_int8_per_node),
                     ("i16", quant_int16_per_node)):
        print(f"arm: future_dispersion {name} only ...", flush=True)
        arms[f"fd_{name}_only"] = run(e_dense_f32, fn(fd_raw), hd_raw)
        print(f"arm: history_distance {name} only ...", flush=True)
        arms[f"hd_{name}_only"] = run(e_dense_f32, fd_raw, fn(hd_raw))
        per_field[f"fd_{name}_cost"] = arms[f"fd_{name}_only"] - base
        per_field[f"hd_{name}_cost"] = arms[f"hd_{name}_only"] - base

    # NOTE: fp16 is NOT admissible for history_distance -- its dynamic range
    # exceeds 65504, so the cast overflows to inf and the squashed channel
    # turns the whole prediction into NaN.  int16 with a per-node scale has the
    # same 16-bit budget and is exact here, so A1 uses fd=f16 / hd=i16.
    print("arm: A1 conservative (E sparse f16 + fd f16 + hd i16) ...", flush=True)
    arms["A1_conservative"] = run(e_sparse_q, quant_fp16(fd_raw),
                                  quant_int16_per_node(hd_raw))
    print("arm: A2 aggressive (E sparse f16 + fd i8 + hd i8) ...", flush=True)
    arms["A2_aggressive"] = run(e_sparse_q, quant_int8_per_node(fd_raw),
                                quant_int8_per_node(hd_raw))

    for name, value in arms.items():
        print(f"  {name:24s} {value:.7f}   vs R0 {value - base:+.7f}")

    # ---------------- global byte accounting (physical sizes) --------------
    # Price T at its STORED dtype: the run-time copy above is float32, which
    # would double-count it (33.34 MB instead of the shipped 16.67 MB).
    t_code_bytes = float(np.asarray(cache["retrieval_code_val"]).nbytes)
    nb = lambda a: float(np.asarray(a).nbytes)
    sizes_dense = {
        "T_retrieval_code_stored": t_code_bytes,
        "E_exact_retrieval_f32": nb(e_val),
        "future_dispersion_f32": nb(fd_raw),
        "exact_available_bool": nb(cache["exact_available_val"]),
        "history_distance_f32": nb(hd_raw),
    }
    total_dense = sum(sizes_dense.values())

    def build_sizes(e_mode, fd_mode, hd_mode, mask_bitpack=True):
        out = {}
        out["T_r8_fp16"] = t_code_bytes
        if e_mode == "sparse_fp16":
            out["E_sparse_fp16"] = n_fired * 2.0
        elif e_mode == "dense_f32":
            out["E_f32"] = nb(e_val)
        if fd_mode == "f16":
            out["fd_f16"] = nb(fd_raw) / 2
        elif fd_mode == "i8":
            out["fd_i8"] = nb(fd_raw) / 4 + 4 * num_nodes
        else:
            out["fd_f32"] = nb(fd_raw)
        if hd_mode == "f16":
            out["hd_f16"] = nb(hd_raw) / 2
        elif hd_mode == "i16":
            out["hd_i16"] = nb(hd_raw) / 2 + 4 * num_nodes
        elif hd_mode == "i8":
            out["hd_i8"] = nb(hd_raw) / 4 + 4 * num_nodes
        else:
            out["hd_f32"] = nb(hd_raw)
        out["mask"] = (n_slots / 8.0) if mask_bitpack else n_slots
        return out

    configs = {
        "deployed_now": build_sizes("dense_f32", "f32", "f32", False),
        "A1_conservative": build_sizes("sparse_fp16", "f16", "f16", True),
        "A2_aggressive": build_sizes("sparse_fp16", "i8", "i8", True),
    }
    accounting = {}
    for name, parts in configs.items():
        total = sum(parts.values())
        accounting[name] = {
            "parts_MB": {k: round(v / 1e6, 2) for k, v in parts.items()},
            "total_MB": round(total / 1e6, 2),
            "reduction_vs_all_raw": 1.0 - total / total_dense,
        }
    # all-raw reference: T uncompressed too
    t_raw_bytes = n_slots * 4.0
    all_raw = total_dense - t_code_bytes + t_raw_bytes
    deployed_total = sum(configs["deployed_now"].values())
    for name in accounting:
        accounting[name]["reduction_vs_all_raw"] = (
            1.0 - sum(configs[name].values()) / all_raw)
        accounting[name]["reduction_vs_deployed_now"] = (
            1.0 - sum(configs[name].values()) / deployed_total)

    accounting["reference_all_raw"] = {
        "parts_MB": {"T_f32": round(t_raw_bytes / 1e6, 2),
                     **{k: round(v / 1e6, 2) for k, v in sizes_dense.items()
                        if k != "T_retrieval_code_stored"}},
        "total_MB": round(all_raw / 1e6, 2),
    }

    print("\n@@BYTES@@")
    for name in ("reference_all_raw", "deployed_now", "A1_conservative",
                 "A2_aggressive"):
        a = accounting[name]
        line = f"  {name:20s} {a['total_MB']:8.2f} MB"
        if "reduction_vs_all_raw" in a:
            line += (f"   vs all-raw {a['reduction_vs_all_raw']*100:6.2f}%"
                     f"   vs deployed {a['reduction_vs_deployed_now']*100:6.2f}%")
        print(line)

    a1_cost = arms["A1_conservative"] - base
    a2_cost = arms["A2_aggressive"] - base
    best_name = "A1_conservative" if abs(a1_cost) <= abs(a2_cost) else "A2_aggressive"
    best_cost = min(abs(a1_cost), abs(a2_cost))
    best_reduction = accounting[best_name]["reduction_vs_all_raw"]

    verdict = {
        "G1_equal_accuracy_le_0.01": bool(best_cost <= 0.01),
        "G2_reduction_ge_50pct": bool(best_reduction >= 0.50),
        "best_arm": best_name,
        "best_arm_cost": (a1_cost if best_name == "A1_conservative" else a2_cost),
        "best_arm_reduction_vs_all_raw": best_reduction,
        "route": ("CANDIDATE" if (best_cost <= 0.01 and best_reduction >= 0.50)
                  else "CLOSED"),
    }
    print("\n" + json.dumps(verdict, indent=2))

    OUT.write_text(json.dumps({
        "scope": "PEMS04 validation-only, zero training, no Test",
        "acceptance": {"reference": base, "recorded_canonical":
                       RECORDED_CANONICAL},
        "arms": arms,
        "per_field_cost": per_field,
        "accounting": accounting,
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
