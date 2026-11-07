"""AIP-TC three-arm end-to-end test (PEMS04, Validation-only, zero training).

Contribution-3 candidate: Asymmetric Identity-Preserving Temporal Compression.
Claim under test: compress the memory PAYLOAD (stored futures/histories) as
rank-8 per-node codes, but keep the DECISION artifacts (which candidates are
selected, the quality scalars) at full precision produced at ingestion time.

Arms
----
C0  Baseline      : deployed frozen candidate index + raw exact values.
T1  AIP-TC        : same index/quality (decision frozen), payload compressed:
                    exact values stored as r8 codes and decoded at runtime.
T2  Naive symm.   : payload compressed AND every decision recomputed online
                    from decoded histories (new top-8, new quality scalars).

T1 and T2 store exactly the same bytes; they differ only in whether decisions
are frozen.  That isolates the value of the decision/payload decoupling.

Pre-registered verdicts (locked before running)
-----------------------------------------------
P1  payload compression is free      : |MAE(T1) - MAE(C0)| <= 0.01
P2  decoupling is necessary          :  MAE(T2) - MAE(T1) >= 0.02
If P2 fails the 40% mis-selection measured by the identity probe has no
downstream cost and the route is closed (reported as such, no packaging).

Protocol: official Validation only, no Test, frozen joint3 checkpoint,
emulator acceptance-tested against the recorded GPU number before comparing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analog_distance import masked_history_distance           # noqa: E402
from compute_analog_retrieval_cache import _training_fallbacks  # noqa: E402
from emulate_and_build_residual_oracle import (                # noqa: E402
    FrozenPipeline, quality_scalars, decode_codes,
)
from utils.datasets_utils import file_path_dict, load_raw_data  # noqa: E402
from run_identity_stability_probe import (                      # noqa: E402
    fit_history_pca, decode, select_topk, CANDIDATE_CACHE,
)

DS = ROOT / "datasets/PEMS04"
EXP = ROOT / (
    "EXP_experiments/TLAST_improved/PEMS04/"
    "TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3"
)
CACHE = DS / "PEMS04_12to12_analog_k8_exact_mean_latc_pca_r8_fp16.npz"
BACKBONE = DS / "PEMS04_12to12_backbone_predictions_joint3.npz"
OUT = Path(__file__).resolve().parent / "aip_tc_three_arm_pems04_val.json"
RECORDED_CANONICAL = 17.8537288  # uniform reranker, 2026-09-20 acceptance run

K = 8
INPUT_LENGTH = 12
MIN_VALID_HISTORY = 6
RANK = 8
X_OFF = np.arange(-11, 1, dtype=np.int64)
Y_OFF = np.arange(1, 13, dtype=np.int64)


def build_arm(raw, tod_by_query, sel_by_query, group_by_query,
              distance_by_query, fallback, scaler_mean, scaler_std,
              num_nodes, num_queries):
    """Replicate compute_analog_retrieval_cache's retrieval + quality scalars
    for an arbitrary candidate selection (one candidate group per query).

    Shapes follow the deployed builder exactly: selected_future is (K, N, H),
    retrieved is (N, H) stored as (H, N), quality distance is the mean over the
    selected candidates' finite distances, dispersion is the standard deviation
    of their futures around the retrieved mean.

    Returns (T_z (S,H,N,1), q0 (S,N), q1 (S,H,N,1)).
    """
    horizon = len(Y_OFF)
    T = np.zeros((num_queries, horizon, num_nodes, 1), dtype=np.float32)
    q0 = np.zeros((num_queries, num_nodes), dtype=np.float32)
    q1 = np.zeros((num_queries, horizon, num_nodes, 1), dtype=np.float32)
    # (1, N): broadcasting with sel (K, N) must not add a leading axis.
    node_index = np.arange(num_nodes)[None, :]
    for qi in range(num_queries):
        sel = sel_by_query[qi]                               # (K, N) positions
        c_group = group_by_query[qi]
        dist_all = distance_by_query[qi]                     # (nc, N)
        c_future = raw[c_group[:, None] + Y_OFF[None], :].astype(
            np.float32, copy=False).transpose(0, 2, 1)       # (nc, N, H)
        selected_future = c_future[sel, node_index, :]       # (K, N, H)
        # Per-query distance matrix has no query axis, so the candidate axis
        # (the one `sel` indexes) is axis 0 here.
        nearest_distance = np.take_along_axis(dist_all, sel, axis=0)  # (K, N)
        selected_valid = (np.isfinite(nearest_distance)[..., None]
                          & (selected_future != 0))
        selected_count = selected_valid.sum(axis=0)          # (N, H)
        selected_sum = np.where(
            selected_valid, selected_future, 0.0
        ).sum(axis=0, dtype=np.float64)
        retrieved = np.zeros_like(selected_sum, dtype=np.float64)
        np.divide(selected_sum, selected_count, out=retrieved,
                  where=selected_count > 0)
        fallback_nh = fallback[int(tod_by_query[qi])].T      # (N, H)
        retrieved = np.where(selected_count > 0, retrieved, fallback_nh)
        T[qi, :, :, 0] = (((retrieved.T - scaler_mean) / scaler_std)
                          .astype(np.float32))
        # quality distance: mean over selected finite distances
        finite_distance = np.isfinite(nearest_distance)
        distance_count = finite_distance.sum(axis=0)
        distance_mean = np.zeros(num_nodes, dtype=np.float64)
        np.divide(np.where(finite_distance, nearest_distance, 0.0).sum(
            axis=0, dtype=np.float64), distance_count, out=distance_mean,
            where=distance_count > 0)
        # NOTE: the deployed builder takes this fallback over the whole TOD
        # block; per query it degenerates to a scalar, so we use the max over
        # the selected candidates (per node).  It is only consulted when no
        # selected candidate has a finite distance, which is rare.
        distance_fallback = np.where(
            finite_distance, nearest_distance, 0.0).max(axis=0)
        distance_mean = np.where(distance_count > 0, distance_mean,
                                 distance_fallback[None, :])
        q0[qi] = distance_mean.astype(np.float32)
        # future dispersion around the retrieved mean
        centered = selected_future - retrieved[None, :, :]
        variance = np.zeros_like(retrieved, dtype=np.float64)
        np.divide(np.where(selected_valid, centered * centered, 0.0).sum(
            axis=0, dtype=np.float64), selected_count, out=variance,
            where=selected_count > 1)
        dispersion = np.sqrt(variance)                       # (N, H)
        dispersion_fallback = dispersion.max(axis=0)
        dispersion = np.where(selected_count > 1, dispersion,
                              dispersion_fallback[None, :])
        q1[qi, :, :, 0] = dispersion.T.astype(np.float32)
    return T, q0, q1


def main():
    global time_data
    index = np.load(DS / "PEMS04_12to12_index_scaler.npz")
    idx_train = np.asarray(index["idx_train"], dtype=np.int64)
    idx_val = np.asarray(index["idx_val"], dtype=np.int64)
    scaler_mean = float(np.asarray(index["scaler_mean"]).item())
    scaler_std = float(np.asarray(index["scaler_std"]).item())

    raw_all, _ = load_raw_data(file_path_dict["PEMS04"])
    raw = np.asarray(raw_all[..., 0], dtype=np.float32)      # (T, N)
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
    t_canonical = decode_codes(cache["retrieval_code_val"], temporal_mean,
                               temporal_basis)

    p_val = np.asarray(np.load(BACKBONE, allow_pickle=True)["backbone_pred_val"],
                       dtype=np.float32)
    y_raw = raw[idx_val[:, None] + Y_OFF[None], :]
    y_z = ((y_raw - scaler_mean) / scaler_std).astype(np.float32)
    valid_q = y_raw != 0

    pipe = FrozenPipeline(EXP, horizon, num_nodes)
    pipe.set_codec(temporal_mean, temporal_basis)

    def mae(pred_z):
        return (float((np.abs(pred_z[..., 0] - y_z) * valid_q).sum()
                      / valid_q.sum()) * scaler_std)

    # ---------- acceptance: reproduce recorded canonical ----------
    base_cache_pred = pipe.forward(p_val, t_canonical, e_val, avail,
                                   q0_c, q1_c, q2_c)
    mae_cache = mae(base_cache_pred)
    print(f"[acceptance] cache-based canonical MAE = {mae_cache:.7f} "
          f"(recorded {RECORDED_CANONICAL})")
    if abs(mae_cache - RECORDED_CANONICAL) > 5e-3:
        raise AssertionError("Emulator does not reproduce recorded benchmark.")

    # ---------- rebuild every arm from raw data, one codec pass each ----------
    query_tod = time_data[idx_val, 0].astype(np.int64)
    train_tod = time_data[idx_train, 0].astype(np.int64)

    with np.load(CANDIDATE_CACHE, allow_pickle=False) as z:
        cached_val = np.asarray(z["candidate_index_val"], dtype=np.int64)

    train_hist = raw[idx_train[:, None] + X_OFF[None], :, ].astype(
        np.float32, copy=False)
    hist_mean, hist_basis, hist_energy = fit_history_pca(train_hist, RANK)
    print(f"history PCA rank {RANK} explained energy = "
          f"{float(hist_energy.mean()):.6f}")

    sel_raw_list, sel_comp_list = [], []
    groups, dist_raw_list, dist_comp_list = [], [], []
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
        c_dec = decode(c_hist, hist_mean, hist_basis, "float16")
        c_dec = np.where(c_hist != 0, c_dec, 0.0)   # validity bitmap stored
        d_comp = masked_history_distance(q_hist, c_dec,
                                         min_valid_history=MIN_VALID_HISTORY,
                                         metric="masked_mse")
        sel_r, nd_r = select_topk(d_raw, K)
        sel_c, nd_c = select_topk(d_comp, K)
        sel_raw_list.append((q_pos, sel_r))
        sel_comp_list.append((q_pos, sel_c))
        groups.append((q_pos, c_group))
        dist_raw_list.append((q_pos, d_raw))
        dist_comp_list.append((q_pos, d_comp))
        # Gate: the raw-distance selection must reproduce the deployed cache,
        # with -1 wherever no candidate reaches min_valid_history.
        raw_selected = np.where(np.isfinite(nd_r), c_group[sel_r], -1)
        if not np.array_equal(raw_selected, cached_val[q_pos]):
            mism = int(np.sum(raw_selected != cached_val[q_pos]))
            raise AssertionError(f"raw selection mismatch vs cache: {mism}")

    def assemble(pairs, distance_pairs):
        """Per-query lists -> (sel_by_query, group_by_query, dist_by_query)."""
        sel_by_q = [None] * len(idx_val)
        grp_by_q = [None] * len(idx_val)
        for (q_pos, sel) in pairs:
            for row, s in zip(q_pos, sel):
                sel_by_q[int(row)] = s
        for (q_pos, group) in groups:
            for row in q_pos:
                grp_by_q[int(row)] = group
        dist_by_q = [None] * len(idx_val)
        for (q_pos, d) in distance_pairs:
            for row, dd in zip(q_pos, d):
                dist_by_q[int(row)] = dd
        return sel_by_q, grp_by_q, dist_by_q

    sel_r_q, grp_q, dist_r_q = assemble(sel_raw_list, dist_raw_list)
    sel_c_q, _, dist_c_q = assemble(sel_comp_list, dist_comp_list)

    tod_by_query = time_data[idx_val, 0].astype(np.int64)
    print("building C0/T1 (frozen decision) arrays ...", flush=True)
    T_frozen, q0_frozen, q1_frozen = build_arm(
        raw, tod_by_query, sel_r_q, grp_q, dist_r_q, fallback, scaler_mean,
        scaler_std, num_nodes, len(idx_val))
    print("building T2 (compressed-domain decision) arrays ...", flush=True)
    T_comp, q0_comp, q1_comp = build_arm(
        raw, tod_by_query, sel_c_q, grp_q, dist_c_q, fallback, scaler_mean,
        scaler_std, num_nodes, len(idx_val))

    def normalise_quality(q0_raw, q1_raw):
        """Mirror quality_scalars(): scale, replicate the query-level distance
        over the horizon, then apply the loader's log1p/(1+·) squash."""
        def squash(x):
            x = np.log1p(np.clip(x, 0.0, None))
            return x / (1.0 + x)
        hd = squash(q0_raw / np.float32(scaler_std ** 2)).astype(np.float32)
        hd = np.repeat(hd[:, None, :], horizon, axis=1)       # (S, H, N)
        fd = squash(q1_raw[..., 0] / np.float32(scaler_std)).astype(np.float32)
        return hd, fd

    q0_frozen_hn, q1_frozen_hn = normalise_quality(q0_frozen, q1_frozen)
    q0_comp_hn, q1_comp_hn = normalise_quality(q0_comp, q1_comp)

    # ---------- C0: frozen decision, raw payload ----------
    pred_c0 = pipe.forward(p_val, T_frozen, e_val, avail,
                           q0_frozen_hn, q1_frozen_hn, q2_c)
    mae_c0 = mae(pred_c0)
    # consistency: rebuilt frozen-decision arm vs cache-based canonical
    print(f"[C0] rebuilt canonical MAE = {mae_c0:.7f} "
          f"(cache-based {mae_cache:.7f}, delta {mae_c0 - mae_cache:+.7f})")

    # ---------- T1: frozen decision, payload compressed (exact values coded)
    e_comp = pipe.codec(e_val)
    pred_t1 = pipe.forward(p_val, T_frozen, e_comp, avail,
                           q0_frozen_hn, q1_frozen_hn, q2_c)
    mae_t1 = mae(pred_t1)

    # ---------- T2: naive symmetric compression ----------
    pred_t2 = pipe.forward(p_val, T_comp, e_comp, avail,
                           q0_comp_hn, q1_comp_hn, q2_c)
    mae_t2 = mae(pred_t2)

    print(f"[T1] AIP-TC      MAE = {mae_t1:.7f}  vs C0 {mae_t1 - mae_c0:+.7f}")
    print(f"[T2] naive symm  MAE = {mae_t2:.7f}  vs C0 {mae_t2 - mae_c0:+.7f}"
          f"   vs T1 {mae_t2 - mae_t1:+.7f}")

    # ---------- storage accounting (per memory entry, per node) ----------
    # full precision: 12-step history + 12-step future, float32
    bytes_full = 12 * 4 + 12 * 4
    # compressed: r8 fp16 codes (8x2 bytes) + 12-bit validity bitmap (2 bytes)
    bytes_compressed = (RANK * 2 + 2) + RANK * 2
    storage = {
        "per_entry_per_node_bytes_full": bytes_full,
        "per_entry_per_node_bytes_compressed": bytes_compressed,
        "reduction": 1.0 - bytes_compressed / bytes_full,
        "note": (
            "T1 and T2 store identical bytes; they differ only in whether the "
            "decision artifacts (candidate index, quality scalars) are frozen "
            "at ingestion or recomputed from decoded histories."
        ),
    }

    # ---------- verdict ----------
    payload_cost = mae_t1 - mae_c0           # cost of compressing the payload
    decision_cost = mae_t2 - mae_t1          # cost of compressed decisions
    verdict = {
        "P1_payload_free": bool(abs(payload_cost) <= 0.01),
        "P2_decoupling_necessary": bool(decision_cost >= 0.02),
        "payload_compression_cost": payload_cost,
        "decision_compression_cost": decision_cost,
        "naive_total_cost": mae_t2 - mae_c0,
        "route": (
            "CANDIDATE"
            if (abs(payload_cost) <= 0.01 and decision_cost >= 0.02)
            else "CLOSED"
        ),
        "thresholds": {"P1_abs": 0.01, "P2_min": 0.02},
    }

    report = {
        "scope": ("PEMS04 validation-only, frozen joint3, zero training, "
                  "no Test; uniform-reranker deployment path"),
        "acceptance": {"cache_based": mae_cache, "recorded": RECORDED_CANONICAL,
                       "rebuilt_c0": mae_c0},
        "arms": {
            "C0_baseline_frozen_decision_raw_payload": mae_c0,
            "T1_aip_tc_frozen_decision_compressed_payload": mae_t1,
            "T2_naive_symmetric_compression": mae_t2,
        },
        "history_pca_explained_energy_rank8": float(hist_energy.mean()),
        "storage": storage,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
