"""Audit + improvement probe for the AIP-TC payload result (PEMS04, val-only).

Two questions:

Q1 (audit)  Is the measured +0.122 for rank-8 exact VALUES numerically
            self-consistent with the reconstruction error those codes incur,
            after dilution by the exact gate (w * avail)?  If the reconstruction
            error is far too small to explain +0.122, the three-arm code is
            wrong.  If it roughly explains it, the number is real.

Q2 (improvement)  Where, if anywhere, can the exact payload still be shrunk?
            (a) sparsity: exact fires only on a subset of slots, so slots with
                avail==0 need no stored value at all (lossless saving).
            (b) residual coding: store the mean field T (already r8) plus a
                low-rank code for (E - T) instead of a low-rank code for E.
            (c) per-node adaptive rank under a fixed average budget.
            (d) quantisation-only (no rank loss): fp16 / int8 on the raw values.

Zero training; validation only.
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

OUT = Path(__file__).resolve().parent / "aip_tc_audit_pems04_val.json"


def fit_field_pca(train_future: np.ndarray, rank: int, min_valid: int):
    """Per-node PCA; returns mean (H,N), basis (N,H,rank), energy (N,)."""
    num_samples, num_time, num_nodes = train_future.shape
    mean = train_future.mean(axis=0)
    basis = np.empty((num_nodes, num_time, rank), dtype=np.float32)
    energy = np.empty((num_nodes,), dtype=np.float64)
    total_var = np.empty((num_nodes,), dtype=np.float64)
    for node in range(num_nodes):
        values = train_future[:, :, node]
        observed = values != 0
        col_mean = mean[:, node][None, :]
        filled = np.where(observed, values, col_mean)
        centered = (filled - col_mean).astype(np.float32)
        keep = observed.sum(axis=1) >= min_valid
        if keep.sum() >= num_time:
            centered = centered[keep]
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        basis[node] = vt[:rank].T.astype(np.float32)
        total = float(np.square(singular).sum())
        energy[node] = (float(np.square(singular[:rank]).sum()) / total
                        if total > 0 else 1.0)
        total_var[node] = total / max(centered.shape[0], 1)
    return mean.astype(np.float32), basis, energy, total_var


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
    pipe.set_codec(None, None, quantize=False)     # no auto codec at all
    w = pipe.w                                     # (H,) exact gate weight
    print("exact gate w per horizon:", np.round(w, 4).tolist(), flush=True)

    def mae(pred_z):
        return (float((np.abs(pred_z[..., 0] - y_z) * valid_q).sum()
                      / valid_q.sum()) * scaler_std)

    # ---------------- Q1: sparsity of the exact channel -------------------
    avail_bin = avail[..., 0] > 0.5                     # (S,H,N)
    slot_total = avail_bin.size
    slot_fired = int(avail_bin.sum())
    fire_rate = slot_fired / slot_total
    print(f"\nexact fire rate: {slot_fired}/{slot_total} = {fire_rate:.4f}")

    # ---------------- rebuild frozen-decision T ---------------------------
    query_tod = time_data[idx_val, 0].astype(np.int64)
    train_tod = time_data[idx_train, 0].astype(np.int64)
    sel_r_q, grp_q, dist_r_q = [None] * len(idx_val), [None] * len(idx_val), \
        [None] * len(idx_val)
    from analog_distance import masked_history_distance
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
    T_frozen, q0_frozen, q1_frozen = build_arm(
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

    def run(T, E):
        return mae(pipe.forward(p_val, T, E, avail, q0_f, q1_f, q2_c))

    base = run(T_frozen, e_val)
    print(f"C0 baseline MAE = {base:.7f}")

    # ---------------- Q1 cont: reconstruction error vs MAE cost ------------
    train_future = raw[idx_train[:, None] + Y_OFF[None], :].astype(
        np.float32, copy=False)
    fut_mean_raw, fut_basis, fut_energy, fut_var = fit_field_pca(
        train_future, RANK, MIN_VALID_HISTORY)
    fut_mean_z = ((fut_mean_raw - scaler_mean) / scaler_std).astype(np.float32)

    def codec_field(values, mean_z, basis):
        centered = values[..., 0] - mean_z[None]
        code = np.einsum("shn,nhr->srn", centered, basis, optimize=True)
        return (mean_z[None] + np.einsum("srn,nhr->shn",
                                         code.astype(np.float32), basis,
                                         optimize=True))[..., None]

    e_c8 = codec_field(e_val, fut_mean_z, fut_basis)
    recon_abs = np.abs(e_c8 - e_val)[..., 0]                 # z space
    # dilution: only slots where exact fires and the label is valid count
    dil = avail_bin & valid_q
    recon_mae_z_fired = float(recon_abs[dil].mean()) if dil.any() else 0.0
    recon_mae_raw_fired = recon_mae_z_fired * scaler_std
    # expected MAE cost ~ mean_h(w) * fire-weighted recon error
    w_mean = float(np.abs(w).mean())
    expected = w_mean * fire_rate * recon_mae_raw_fired
    print(f"\nrecon |E_codec - E| over fired slots = {recon_mae_z_fired:.6f} z "
          f"= {recon_mae_raw_fired:.5f} raw")
    print(f"mean|w| = {w_mean:.4f}, fire_rate = {fire_rate:.4f}")
    print(f"naive expected MAE cost  ~= {expected:.5f}")
    mae_e8 = run(T_frozen, e_c8)
    print(f"measured MAE cost        = {mae_e8 - base:+.5f}")
    print(f"ratio measured/expected  = "
          f"{(mae_e8 - base) / expected if expected else float('nan'):.2f}")

    # ---------------- Q2a: sparsity -- drop stored values where avail==0 ---
    e_sparse = np.where(avail > 0.5, e_val, fut_mean_z[None, ..., None])
    mae_sparse_fallback = run(T_frozen, e_sparse)
    print(f"\n[Q2a] zero-out unfired slots (store nothing there, fall back to "
          f"mean): {mae_sparse_fallback - base:+.7f}")
    storage_saving_sparse = 1.0 - fire_rate
    print(f"      lossless storage saving on E = {storage_saving_sparse:.4f}")

    # ---------------- Q2b: residual coding E - T ---------------------------
    resid_train = None
    # build the residual field on TRAIN: exact value minus mean retrieval,
    # using the train split's own cache entries.
    e_train = np.asarray(cache["exact_retrieval_train"], dtype=np.float32)
    # approximate the train mean-retrieval T by the retrieval_temporal_mean-free
    # reconstruction from the stored codes (that IS the deployed T)
    tm = np.asarray(cache["retrieval_temporal_mean"], dtype=np.float32)
    tb = np.asarray(cache["retrieval_temporal_basis"], dtype=np.float32)
    tcode_train = np.asarray(cache["retrieval_code_train"], dtype=np.float32)
    T_train = tm[None] + np.einsum("srnc,nhr->shnc", tcode_train, tb,
                                   optimize=True)
    resid_train = (e_train - T_train)[..., 0]                # (S,H,N) z space
    avail_train = np.asarray(cache["exact_available_train"],
                             dtype=np.float32)[..., 0] > 0.5
    # fit PCA only on fired slots of the residual field, per node
    rank = RANK
    r_basis = np.empty((num_nodes, horizon, rank), dtype=np.float32)
    r_mean = np.zeros((horizon, num_nodes), dtype=np.float32)
    r_energy = np.empty((num_nodes,), dtype=np.float64)
    for node in range(num_nodes):
        col = resid_train[:, :, node]
        mask = avail_train[:, :, node]
        if mask.sum() < horizon:
            r_basis[node] = fut_basis[node]
            r_energy[node] = 0.0
            continue
        r_mean[:, node] = col[mask].mean(axis=0)
        centered = np.where(mask, col - r_mean[:, node][None, :], 0.0)
        centered = centered[mask.any(axis=1)]
        if centered.shape[0] < horizon:
            r_basis[node] = fut_basis[node]
            r_energy[node] = 0.0
            continue
        _, s, vt = np.linalg.svd(centered.astype(np.float32),
                                 full_matrices=False)
        r_basis[node] = vt[:rank].T.astype(np.float32)
        tot = float(np.square(s).sum())
        r_energy[node] = float(np.square(s[:rank]).sum()) / tot if tot else 1.0

    # comparison: energy of E field vs (E-T) field at the same rank
    e_field_fired = e_val[..., 0]
    print(f"\n[Q2b] residual coding (E - T)")
    print(f"      explained energy rank{rank}: E field = "
          f"{float(fut_energy.mean()):.6f}  |  (E-T) field = "
          f"{float(r_energy.mean()):.6f}")

    def codec_resid(values, mean_z, basis):
        centered = values[..., 0] - mean_z[None]
        code = np.einsum("shn,nhr->srn", centered, basis, optimize=True)
        return (mean_z[None] + np.einsum("srn,nhr->shn",
                                         code.astype(np.float32), basis,
                                         optimize=True))[..., None]

    # T as the deployed r8 code gives; E = T_r8 + coded residual
    tcode_val = np.asarray(cache["retrieval_code_val"], dtype=np.float32)
    T_val_r8 = tm[None] + np.einsum("srnc,nhr->shnc", tcode_val, tb,
                                    optimize=True)
    resid_val = (e_val - T_val_r8)[..., 0]
    e_resid = (T_val_r8 + codec_resid(resid_val[..., None], r_mean, r_basis))
    mae_resid = run(T_val_r8, e_resid)
    print(f"      MAE with T=deployed r8, E = T_r8 + rank{rank}(E-T) "
          f"residual: {mae_resid - base:+.7f}")
    print(f"      (vs direct rank{rank} code of E: {run(T_val_r8, codec_field(e_val, fut_mean_z, fut_basis)) - base:+.7f})")

    # ---------------- Q2d: quantisation only, no rank loss ----------------
    def quant_only(values, dtype):
        return values.astype(dtype).astype(np.float32)

    for dt in (np.float16,):
        m = run(T_frozen, quant_only(e_val, dt))
        print(f"\n[Q2d] E quantised to {np.dtype(dt).name} only (rank kept): "
              f"{m - base:+.7f}")

    # int8 with per-node scale on the raw-domain-equivalent z values
    lo = float(np.percentile(e_val, 0.1))
    hi = float(np.percentile(e_val, 99.9))
    scale = (hi - lo) / 254.0
    q = np.clip(np.round((e_val - lo) / scale), 0, 254)
    e_int8 = (q * scale + lo).astype(np.float32)
    m_int8 = run(T_frozen, e_int8)
    print(f"[Q2d] E int8 (per-tensor scale, 8 bit): {m_int8 - base:+.7f}")
    for bits in (6, 5, 4):
        levels = 2 ** bits - 1
        sc = (hi - lo) / levels
        qq = np.clip(np.round((e_val - lo) / sc), 0, levels)
        m_b = run(T_frozen, (qq * sc + lo).astype(np.float32))
        print(f"[Q2d] E int{bits}: {m_b - base:+.7f}")

    report = {
        "scope": "PEMS04 validation-only, zero training, no Test",
        "baseline_C0_mae": base,
        "exact_gate_w": [float(x) for x in w],
        "exact_fire_rate": fire_rate,
        "recon_mae_z_over_fired_slots": recon_mae_z_fired,
        "recon_mae_raw_over_fired_slots": recon_mae_raw_fired,
        "naive_expected_cost": expected,
        "measured_cost_rank8_E": mae_e8 - base,
        "Q2a_sparse_storage": {
            "mae_delta": mae_sparse_fallback - base,
            "lossless_storage_saving_on_E": storage_saving_sparse,
        },
        "Q2b_residual_coding": {
            "energy_E_field_rank8": float(fut_energy.mean()),
            "energy_residual_field_rank8": float(r_energy.mean()),
            "mae_delta": mae_resid - base,
        },
        "Q2d_quantisation": {
            "fp16": mae_e8 * 0 + (run(T_frozen, quant_only(e_val, np.float16))
                                  - base),
            "int8": m_int8 - base,
        },
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
