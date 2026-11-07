"""(PEMS04, validation-only, zero training.)

Findings that motivate this script (from run_aip_tc_audit.py):
  * exact fires on only 9.48% of (query, horizon, node) slots, so 90.5% of the
    stored exact payload is provably dead weight (delta = 0.0000000).
  * low-rank r8 on the exact VALUES costs +0.122 (real, not a bug).
  * fp16 quantisation of the exact values, rank preserved, costs +0.0009.

So the payload should be shrunk by sparsity + quantisation, NOT by low rank.
This script measures the combined arm and prices the total memory state.

Arms (all with the frozen raw-domain decision, T given as the deployed r8 code):
  C0   T = deployed r8,  E = raw float32 everywhere          (reference)
  S    E stored only where exact fires, fp32                 (sparse)
  SQ   E stored only where exact fires, fp16                 (sparse + quant)
  R8   E low-rank r8 everywhere                              (AIP-TC payload)
  SR8  E low-rank r8, stored only where exact fires          (best effort)

Pre-registered criterion: the combined arm must cost <= 0.01 MAE vs C0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_analog_retrieval_cache import _training_fallbacks      # noqa: E402
from emulate_and_build_residual_oracle import (                     # noqa: E402
    FrozenPipeline, quality_scalars,
)
from utils.datasets_utils import file_path_dict, load_raw_data      # noqa: E402
from run_identity_stability_probe import select_topk                # noqa: E402
from run_aip_tc_three_arm import (                                  # noqa: E402
    build_arm, DS, EXP, CACHE, BACKBONE, K, MIN_VALID_HISTORY, RANK,
    X_OFF, Y_OFF,
)
from analog_distance import masked_history_distance                 # noqa: E402

OUT = Path(__file__).resolve().parent / "aip_tc_sparse_pems04_val.json"


def fit_field_pca(train_future, rank, min_valid):
    n_samp, n_t, n_node = train_future.shape
    mean = train_future.mean(axis=0)
    basis = np.empty((n_node, n_t, rank), dtype=np.float32)
    for node in range(n_node):
        values = train_future[:, :, node]
        observed = values != 0
        col_mean = mean[:, node][None, :]
        centered = (np.where(observed, values, col_mean)
                    - col_mean).astype(np.float32)
        keep = observed.sum(axis=1) >= min_valid
        if keep.sum() >= n_t:
            centered = centered[keep]
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis[node] = vt[:rank].T.astype(np.float32)
    return mean.astype(np.float32), basis


def main():
    index = np.load(DS / "PEMS04_12to12_index_scaler.npz")
    idx_train = np.asarray(index["idx_train"], dtype=np.int64)
    idx_val = np.asarray(index["idx_val"], dtype=np.int64)
    scaler_mean = float(np.asarray(index["scaler_mean"]).item())
    scaler_std = float(np.asarray(index["scaler_std"]).item())

    raw_all, _ = load_raw_data(file_path_dict["PEMS04"])
    raw = np.asarray(raw_all[..., 0], dtype=np.float32)
    time_data = np.load(DS / "PEMS04_time.npz", allow_pickle=False)["time"]
    horizon, num_nodes = 12, raw.shape[1]
    num_tod = int(time_data[:, 0].max()) + 1
    fallback = _training_fallbacks(raw_all[..., :1], time_data, idx_train,
                                   Y_OFF, num_tod)

    cache = np.load(CACHE, allow_pickle=True)
    e_val = np.asarray(cache["exact_retrieval_val"], dtype=np.float32)
    avail = np.asarray(cache["exact_available_val"], dtype=np.float32)
    q0_c, q1_c, q2_c = quality_scalars(cache, scaler_std, horizon)
    p_val = np.asarray(np.load(BACKBONE, allow_pickle=True)["backbone_pred_val"],
                       dtype=np.float32)
    y_raw = raw[idx_val[:, None] + Y_OFF[None], :]
    y_z = ((y_raw - scaler_mean) / scaler_std).astype(np.float32)
    valid_q = y_raw != 0

    pipe = FrozenPipeline(EXP, horizon, num_nodes)
    pipe.set_codec(None, None, quantize=False)

    def mae(pred_z):
        return (float((np.abs(pred_z[..., 0] - y_z) * valid_q).sum()
                      / valid_q.sum()) * scaler_std)

    # frozen raw-domain decision -> T (uncompressed) and deployed r8 T
    query_tod = time_data[idx_val, 0].astype(np.int64)
    train_tod = time_data[idx_train, 0].astype(np.int64)
    sel_r_q, grp_q, dist_r_q = [None] * len(idx_val), [None] * len(idx_val), \
        [None] * len(idx_val)
    for tod in np.unique(query_tod):
        q_pos = np.flatnonzero(query_tod == tod)
        c_group = idx_train[train_tod == tod]
        q_hist = raw[idx_val[q_pos][:, None] + X_OFF[None], :, ].astype(
            np.float32, copy=False)
        c_hist = raw[c_group[:, None] + X_OFF[None], :, ].astype(
            np.float32, copy=False)
        d_raw = masked_history_distance(q_hist, c_hist,
                                        min_valid_history=MIN_VALID_HISTORY,
                                        metric="masked_mse")
        sel_r, _ = select_topk(d_raw, K)
        for row, s, d in zip(q_pos, sel_r, d_raw):
            row = int(row)
            sel_r_q[row], grp_q[row], dist_r_q[row] = s, c_group, d

    tod_by_query = time_data[idx_val, 0].astype(np.int64)
    T_raw, q0_frozen, q1_frozen = build_arm(
        raw, tod_by_query, sel_r_q, grp_q, dist_r_q, fallback, scaler_mean,
        scaler_std, num_nodes, len(idx_val))

    def normalise(q0_raw, q1_raw):
        def squash(x):
            x = np.log1p(np.clip(x, 0.0, None))
            return x / (1.0 + x)
        hd = np.repeat(squash(q0_raw / np.float32(scaler_std ** 2)).astype(
            np.float32)[:, None, :], horizon, axis=1)
        fd = squash(q1_raw[..., 0] / np.float32(scaler_std)).astype(np.float32)
        return hd, fd

    q0_f, q1_f = normalise(q0_frozen, q1_frozen)

    tm = np.asarray(cache["retrieval_temporal_mean"], dtype=np.float32)
    tb = np.asarray(cache["retrieval_temporal_basis"], dtype=np.float32)
    tcode_val = np.asarray(cache["retrieval_code_val"], dtype=np.float32)
    T_r8 = tm[None] + np.einsum("srnc,nhr->shnc", tcode_val, tb, optimize=True)

    def run(T, E):
        return mae(pipe.forward(p_val, T, E, avail, q0_f, q1_f, q2_c))

    fired = avail > 0.5

    train_future = raw[idx_train[:, None] + Y_OFF[None], :].astype(
        np.float32, copy=False)
    fut_mean_raw, fut_basis = fit_field_pca(train_future, RANK,
                                            MIN_VALID_HISTORY)
    fut_mean_z = ((fut_mean_raw - scaler_mean) / scaler_std).astype(np.float32)

    def codec_field(values, mean_z, basis):
        centered = values[..., 0] - mean_z[None]
        code = np.einsum("shn,nhr->srn", centered, basis, optimize=True)
        return (mean_z[None] + np.einsum("srn,nhr->shn",
                                         code.astype(np.float32), basis,
                                         optimize=True))[..., None]

    arms = {}
    arms["C0_T_r8_E_f32_dense"] = run(T_r8, e_val)
    arms["C0b_T_raw_E_f32_dense"] = run(T_raw, e_val)

    # S: sparse fp32 -- unfired slots carry no value (mean fallback is inert)
    e_sparse = np.where(fired, e_val, fut_mean_z[None, ..., None].astype(
        np.float32))
    arms["S_sparse_f32"] = run(T_r8, e_sparse)

    # SQ: sparse + fp16 on the stored (fired) entries only
    e_sparse_q = np.where(fired, e_val.astype(np.float16).astype(np.float32),
                          fut_mean_z[None, ..., None].astype(np.float32))
    arms["SQ_sparse_fp16"] = run(T_r8, e_sparse_q)

    # R8 / SR8: low-rank on the exact values (the AIP-TC payload hypothesis)
    e_r8 = codec_field(e_val, fut_mean_z, fut_basis)
    arms["R8_dense"] = run(T_r8, e_r8)
    arms["SR8_sparse"] = run(T_r8, np.where(fired, e_r8, 0.0))

    base = arms["C0_T_r8_E_f32_dense"]
    print(f"\nreference C0 (T = deployed r8, E = raw fp32 dense) = {base:.7f}")
    for name, value in arms.items():
        print(f"  {name:26s} {value:.7f}   vs C0 {value - base:+.7f}")

    # ---------------- storage accounting ---------------------------------
    n_slots = int(fired.size)
    n_fired = int(fired.sum())
    fire_rate = n_fired / n_slots
    # bits per stored element
    BITS = {"f32": 32.0, "f16": 16.0}
    dense_f32 = n_slots * BITS["f32"]
    sparse_f32 = n_fired * BITS["f32"] + n_slots * 1.0        # +1 bit bitmap
    sparse_f16 = n_fired * BITS["f16"] + n_slots * 1.0
    r8_dense = n_slots * RANK * BITS["f32"] / horizon          # 8 codes / 12 steps
    accounting = {
        "slots": n_slots,
        "fired_slots": n_fired,
        "fire_rate": fire_rate,
        "bits_per_element": {
            "E_dense_f32": dense_f32 / n_slots,
            "E_sparse_f32_plus_bitmap": sparse_f32 / n_slots,
            "E_sparse_f16_plus_bitmap": sparse_f16 / n_slots,
            "T_deployed_r8_f16": BITS["f16"] * RANK / horizon,
        },
        "E_storage_saving_vs_dense_f32": {
            "sparse_f32": 1.0 - sparse_f32 / dense_f32,
            "sparse_f16": 1.0 - sparse_f16 / dense_f32,
        },
        "note": ("T is priced as the deployed fp16 r8 code (8 codes per 12 "
                 "steps per node); the bitmap is priced at 1 bit per slot."),
    }
    print("\nstorage accounting (E channel, bits per (query,horizon,node) "
          "slot):")
    for k, v in accounting["bits_per_element"].items():
        print(f"  {k:30s} {v:8.3f}")
    for k, v in accounting["E_storage_saving_vs_dense_f32"].items():
        print(f"  saving {k:24s} {v * 100:6.2f}%")

    # ---------------- how much does the exact channel contribute at all? ---
    zero_avail = np.zeros_like(avail)
    mae_no_exact = mae(pipe.forward(p_val, T_r8, e_val, zero_avail,
                                    q0_f, q1_f, q2_c))
    exact_gain = mae_no_exact - base
    contribution = {
        "mae_without_exact_channel": mae_no_exact,
        "exact_channel_total_gain": exact_gain,
        "fire_rate": fire_rate,
        "gain_per_fired_slot": (exact_gain / fire_rate
                                if fire_rate else float("nan")),
        "note": ("gain_per_fired_slot is the MAE-equivalent contribution "
                 "attributable to the exact channel divided by the fraction "
                 "of slots on which it fires; it is an aggregate density, "
                 "not a per-slot causal estimate."),
    }
    print(f"\nexact channel: off -> {mae_no_exact:.7f}  "
          f"(gain {exact_gain:+.7f})  fire rate {fire_rate:.4f}  "
          f"gain/fire_rate = {contribution['gain_per_fired_slot']:.4f}")

    arms["X_exact_channel_off"] = mae_no_exact

    best = min(arms["S_sparse_f32"], arms["SQ_sparse_fp16"])
    verdict = {
        "criterion_MAE_cost_le_0.01": bool(abs(best - base) <= 0.01),
        "best_arm_cost": best - base,
        "low_rank_cost": arms["R8_dense"] - base,
        "sparse_beats_low_rank_by": (arms["R8_dense"] - best),
        "route": ("CANDIDATE" if abs(best - base) <= 0.01 else "CLOSED"),
    }
    print("\n" + json.dumps(verdict, indent=2))

    OUT.write_text(json.dumps({
        "scope": "PEMS04 validation-only, zero training, no Test",
        "arms": arms,
        "accounting": accounting,
        "exact_channel_contribution": contribution,
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
