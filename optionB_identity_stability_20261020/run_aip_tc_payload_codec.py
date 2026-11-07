"""AIP-TC payload decomposition (PEMS04, validation-only, zero training).

The three-arm test showed that re-using the DEPLOYED basis (fit on K-candidate
MEANS) to compress the exact-match VALUES costs +0.118 MAE.  This script
separates two confounded things:

  (i)  what the codec is fit on  -- candidate means (deployed) vs individual
       candidate futures (what the exact payload actually is);
  (ii) which field is compressed -- the mean retrieval T, the exact values E,
       or both.

It also re-checks the decision/payload decoupling (P2) under the best payload
codec, because the three-arm test measured it on top of a badly-codec'd payload.

Pipeline is run with the auto-codec disabled so every arm is coded exactly once
by the codec named in its label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analog_distance import masked_history_distance              # noqa: E402
from compute_analog_retrieval_cache import _training_fallbacks    # noqa: E402
from emulate_and_build_residual_oracle import (                   # noqa: E402
    FrozenPipeline, quality_scalars, decode_codes,
)
from utils.datasets_utils import file_path_dict, load_raw_data    # noqa: E402
from run_identity_stability_probe import select_topk              # noqa: E402
from run_aip_tc_three_arm import (                                # noqa: E402
    build_arm, DS, EXP, CACHE, BACKBONE, K, MIN_VALID_HISTORY, RANK,
    X_OFF, Y_OFF,
)

OUT = Path(__file__).resolve().parent / "aip_tc_payload_codec_pems04_val.json"


def fit_futures_pca(train_future: np.ndarray, rank: int, min_valid: int):
    """Per-node PCA on INDIVIDUAL candidate futures (not their mean).

    train_future: (sample, horizon, node).  Returns mean (horizon, node),
    basis (node, horizon, rank), energy (node,).
    """
    num_samples, num_time, num_nodes = train_future.shape
    mean = train_future.mean(axis=0)
    basis = np.empty((num_nodes, num_time, rank), dtype=np.float32)
    energy = np.empty((num_nodes,), dtype=np.float64)
    for node in range(num_nodes):
        values = train_future[:, :, node]
        observed = values != 0
        filled = np.where(observed, values, mean[:, node][None, :])
        centered = (filled - mean[:, node][None, :]).astype(np.float32)
        keep = observed.sum(axis=1) >= min_valid
        if keep.sum() >= num_time:
            centered = centered[keep]
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        basis[node] = vt[:rank].T.astype(np.float32)
        total = float(np.square(singular).sum())
        energy[node] = (float(np.square(singular[:rank]).sum()) / total
                        if total > 0 else 1.0)
    return mean.astype(np.float32), basis, energy


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
    temporal_mean = np.asarray(cache["retrieval_temporal_mean"], dtype=np.float32)
    temporal_basis = np.asarray(cache["retrieval_temporal_basis"], dtype=np.float32)
    e_val = np.asarray(cache["exact_retrieval_val"], dtype=np.float32)
    avail = np.asarray(cache["exact_available_val"], dtype=np.float32)
    q0_c, q1_c, q2_c = quality_scalars(cache, scaler_std, horizon)

    p_val = np.asarray(np.load(BACKBONE, allow_pickle=True)["backbone_pred_val"],
                       dtype=np.float32)
    y_raw = raw[idx_val[:, None] + Y_OFF[None], :]
    y_z = ((y_raw - scaler_mean) / scaler_std).astype(np.float32)
    valid_q = y_raw != 0

    # auto-codec disabled: every arm is coded exactly once by its own codec
    pipe = FrozenPipeline(EXP, horizon, num_nodes)
    pipe.set_codec(temporal_mean, temporal_basis, quantize=False)

    def mae(pred_z):
        return (float((np.abs(pred_z[..., 0] - y_z) * valid_q).sum()
                      / valid_q.sum()) * scaler_std)

    # ---------- codecs ----------
    train_future = raw[idx_train[:, None] + Y_OFF[None], :].astype(
        np.float32, copy=False)
    fut_mean_raw, fut_basis, fut_energy = fit_futures_pca(
        train_future, RANK, MIN_VALID_HISTORY)
    # z-space equivalent of the raw-space futures mean
    fut_mean_z = ((fut_mean_raw - scaler_mean) / scaler_std).astype(np.float32)

    def codec_field(values, mean_z, basis):
        """values (S,H,N,1) in z-space -> rank-r fp16 round trip."""
        centered = values[..., 0] - mean_z[None]                 # (S,H,N)
        code = np.einsum("shn,nhr->srn", centered, basis,
                         optimize=True).astype(np.float16)
        return (mean_z[None] + np.einsum("srn,nhr->shn",
                                         code.astype(np.float32), basis,
                                         optimize=True))[..., None]

    def codec_deployed(values):
        """The deployed codec: fp16 codes on the mean-fit basis."""
        centered = values[..., 0] - temporal_mean[..., 0][None]
        code = np.einsum("shn,nhr->srn", centered,
                         temporal_basis, optimize=True).astype(np.float16)
        return (temporal_mean[..., 0][None] + np.einsum(
            "srn,nhr->shn", code.astype(np.float32), temporal_basis,
            optimize=True))[..., None]

    print(f"futures PCA rank {RANK} explained energy = "
          f"{float(fut_energy.mean()):.6f} "
          f"(deployed mean-field basis was fit on K-candidate means)")

    # ---------- rebuild T (frozen decision) and the compressed-domain decision
    query_tod = time_data[idx_val, 0].astype(np.int64)
    train_tod = time_data[idx_train, 0].astype(np.int64)
    hist_mean, hist_basis = None, None
    from run_identity_stability_probe import fit_history_pca
    train_hist = raw[idx_train[:, None] + X_OFF[None], :, ].astype(
        np.float32, copy=False)
    hist_mean, hist_basis, hist_energy = fit_history_pca(train_hist, RANK)

    sel_r_q, grp_q, dist_r_q = [None] * len(idx_val), [None] * len(idx_val), \
        [None] * len(idx_val)
    sel_c_q, dist_c_q = [None] * len(idx_val), [None] * len(idx_val)
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
        c_dec = np.where(
            c_hist != 0,
            (hist_mean[None] + np.einsum(
                "srn,ntr->stn",
                np.einsum("stn,ntr->srn",
                          (np.where(c_hist != 0, c_hist, hist_mean[None])
                           - hist_mean[None]), hist_basis,
                          optimize=True).astype(np.float16).astype(np.float32),
                hist_basis, optimize=True)), 0.0)
        d_comp = masked_history_distance(q_hist, c_dec,
                                         min_valid_history=MIN_VALID_HISTORY,
                                         metric="masked_mse")
        sel_r, _ = select_topk(d_raw, K)
        sel_c, _ = select_topk(d_comp, K)
        for row, s, c_s, d, d_c in zip(q_pos, sel_r, sel_c, d_raw, d_comp):
            row = int(row)
            sel_r_q[row], sel_c_q[row] = s, c_s
            grp_q[row], dist_r_q[row], dist_c_q[row] = c_group, d, d_c

    tod_by_query = time_data[idx_val, 0].astype(np.int64)
    T_frozen, q0_frozen, q1_frozen = build_arm(
        raw, tod_by_query, sel_r_q, grp_q, dist_r_q, fallback, scaler_mean,
        scaler_std, num_nodes, len(idx_val))
    T_comp, q0_comp, q1_comp = build_arm(
        raw, tod_by_query, sel_c_q, grp_q, dist_c_q, fallback, scaler_mean,
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
    q0_c2, q1_c2 = normalise(q0_comp, q1_comp)

    def run(T, E, q0n, q1n):
        return mae(pipe.forward(p_val, T, E, avail, q0n, q1n, q2_c))

    arms = {}
    arms["C0_T_raw_E_raw"] = run(T_frozen, e_val, q0_f, q1_f)
    arms["E_deployed_basis"] = run(T_frozen, codec_deployed(e_val), q0_f, q1_f)
    arms["E_futures_basis"] = run(
        T_frozen, codec_field(e_val, fut_mean_z, fut_basis), q0_f, q1_f)
    arms["T_futures_basis"] = run(
        codec_field(T_frozen, fut_mean_z, fut_basis), e_val, q0_f, q1_f)
    arms["T_and_E_futures_basis"] = run(
        codec_field(T_frozen, fut_mean_z, fut_basis),
        codec_field(e_val, fut_mean_z, fut_basis), q0_f, q1_f)
    arms["T2_full_futures_basis"] = run(
        codec_field(T_comp, fut_mean_z, fut_basis),
        codec_field(e_val, fut_mean_z, fut_basis), q0_c2, q1_c2)

    base = arms["C0_T_raw_E_raw"]
    print(f"\nbaseline C0 (T raw, E raw) MAE = {base:.7f}")
    for name, value in arms.items():
        print(f"  {name:28s} MAE = {value:.7f}  vs C0 {value - base:+.7f}")

    # ---------- sanity sweep: does the exact-value cost come from rank
    # truncation?  Rank 12 is lossless, so its cost must be ~0.
    sweep = {}
    for r in (12, 10, 8, 6, 4):
        _, basis_r, energy_r = fit_futures_pca(train_future, r,
                                               MIN_VALID_HISTORY)
        value = run(T_frozen, codec_field(e_val, fut_mean_z, basis_r),
                    q0_f, q1_f)
        sweep[f"E_rank{r}"] = {
            "mae": value, "delta_vs_C0": value - base,
            "explained_energy": float(energy_r.mean()),
        }
        print(f"  sanity E_rank{r:<3d} MAE = {value:.7f}  "
              f"vs C0 {value - base:+.7f}  energy "
              f"{float(energy_r.mean()):.6f}")

    payload_best = min(arms["E_futures_basis"], arms["T_and_E_futures_basis"])
    decision_cost = arms["T2_full_futures_basis"] - arms["T_and_E_futures_basis"]
    verdict = {
        "P1_payload_free_best_case": bool(abs(payload_best - base) <= 0.01),
        "payload_cost_best_case": payload_best - base,
        "P2_decision_cost_under_best_payload": decision_cost,
        "P2_decoupling_necessary": bool(decision_cost >= 0.02),
        "route": (
            "CANDIDATE"
            if (abs(payload_best - base) <= 0.01 and decision_cost >= 0.02)
            else "CLOSED"
        ),
        "thresholds": {"P1_abs": 0.01, "P2_min": 0.02},
    }
    report = {
        "scope": ("PEMS04 validation-only, frozen joint3, zero training, no "
                  "Test; codec applied exactly once per arm"),
        "futures_pca_explained_energy_rank8": float(fut_energy.mean()),
        "history_pca_explained_energy_rank8": float(hist_energy.mean()),
        "arms": arms,
        "exact_value_rank_sweep": sweep,
        "baseline": "C0_T_raw_E_raw",
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
