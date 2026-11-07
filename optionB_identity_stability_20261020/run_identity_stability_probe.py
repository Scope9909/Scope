"""Zero-training probe: retrieval-identity stability under temporal compression.

Question (Option B / memory-axis material)
------------------------------------------
If the memory's stored HISTORIES were compressed to per-node rank-r temporal
PCA codes (the same family of compression already deployed for the stored
FUTURES in ``..._latc_pca_r8_fp16.npz``), and candidate ranking ran against
the decoded reconstructions, would the deployed top-8 retrieval identity
survive?

The deployed PCA-r8 cache compresses only the stored retrieval *values*;
candidate selection is frozen and inherited, so identity is unchanged *by
construction* there.  The unmeasured question is whether the *selection rule
itself* is stable in the low-rank temporal subspace.  This probe answers it
without any training.

Pre-registered protocol
-----------------------
- Split: validation only.  Test is not touched.
- PCA fit: per node, on TRAIN endpoint histories only (leakage-safe, mirrors
  ``retrieval_compression_fit_split='train'``).  Missing (zero) history
  points are imputed by the per-timestep train mean before projection, the
  same convention as the deployed futures PCA.
- Primary variant ``mem_r{r}``: query history raw (live sensor), candidate
  histories decoded from rank-r codes (deployment-realistic).
- Secondary variants: ``both_r8`` (query side also decoded), ``mem_r8_fp16``
  (codes rounded to float16 before decoding, matching deployed code dtype).
- Rank curve: 12 (lossless control, must reproduce identity ~exactly),
  8 (primary), 6, 4.
- Correctness gate: a from-scratch raw-distance re-selection must reproduce
  the deployed ``candidate_index_val`` bit-exactly before any comparison is
  reported.  If the gate fails the probe aborts with no results.

Metrics (over val queries x nodes)
----------------------------------
- top1_agreement               : fraction where rank-1 endpoint is identical
- top8_overlap                 : mean |raw_top8 ∩ comp_top8| / K
- tie_fraction                 : fraction where raw d(rank1) == d(rank2)
- top1_agreement_untied        : top1 agreement restricted to untied slots
- zerodist_any_retention       : among slots whose raw top-8 contains >=1
                                 zero-distance candidate, fraction where the
                                 compressed top-8 still contains >=1 of them
- zerodist_set_retention       : mean fraction of the raw zero-distance
                                 endpoint set retained in compressed top-8
- equal_distance_swaps         : among changed rank-1 slots, count where the
                                 replacement's raw distance equals the
                                 original's (semantically neutral swap)
- replacement_excess_median    : median of (d_raw(new) - d_raw(old)) over
                                 changed rank-1 slots, in raw masked-MSE units

Pre-registered reading
----------------------
- top1 >= 0.95 AND top8_overlap >= 0.98 AND zerodist_any_retention >= 0.99
  -> identity stable; supports the "retrieval lives in a low-rank temporal
  subspace" mechanism claim for the memory axis.
- Otherwise the compression boundary is reported honestly; the memory-axis
  claim stays limited to value-side compression (already measured).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analog_distance import masked_history_distance  # noqa: E402
from utils.datasets_utils import file_path_dict, load_raw_data  # noqa: E402

DATASET = "PEMS04"
K = 8
INPUT_LENGTH = 12
MIN_VALID_HISTORY = 6
RANKS = (12, 8, 6, 4)

CANDIDATE_CACHE = ROOT / "datasets" / DATASET / (
    f"{DATASET}_12to12_analog_k8_candidate_indices.npz"
)
FUTURES_PCA_CACHE = ROOT / "datasets" / DATASET / (
    f"{DATASET}_12to12_analog_k8_exact_mean_latc_pca_r8_fp16.npz"
)
OUTPUT = Path(__file__).resolve().parent / "identity_stability_pems04_val.json"


def select_topk(distance: np.ndarray, k: int):
    """Replicates the deployed 'nearest' selection: argpartition top-k then a
    stable ascending sort (identical to compute_analog_candidate_index_cache).
    """
    nearest = np.argpartition(distance, kth=k - 1, axis=1)[:, :k, :]
    nearest_distance = np.take_along_axis(distance, nearest, axis=1)
    order = np.argsort(nearest_distance, axis=1, kind="stable")
    nearest = np.take_along_axis(nearest, order, axis=1)
    return nearest, np.take_along_axis(nearest_distance, order, axis=1)


def fit_history_pca(train_hist: np.ndarray, rank: int):
    """Per-node PCA on train histories.

    ``train_hist`` is (sample, time, node).  Returns (mean, basis, energy)
    with mean (time, node) and basis (node, time, rank) whose columns are
    orthonormal, matching the deployed futures-PCA layout.
    """
    num_samples, num_time, num_nodes = train_hist.shape
    mean = train_hist.mean(axis=0)  # (time, node)
    basis = np.empty((num_nodes, num_time, min(rank, num_time)), dtype=np.float32)
    energy = np.empty((num_nodes,), dtype=np.float64)
    for node in range(num_nodes):
        values = train_hist[:, :, node]
        observed = values != 0
        # Missing points imputed by the per-timestep train mean, matching the
        # deployed futures-PCA convention, then centred.
        filled = np.where(observed, values, mean[:, node][None, :])
        centered = (filled - mean[:, node][None, :]).astype(np.float32)
        # Samples with too little observation carry no reliable shape signal;
        # drop them from the fit when enough samples remain.
        keep = observed.sum(axis=1) >= MIN_VALID_HISTORY
        if keep.sum() >= num_time:
            centered = centered[keep]
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        basis[node] = vt[:rank].T.astype(np.float32)
        total = float(np.square(singular).sum())
        energy[node] = (
            float(np.square(singular[:rank]).sum()) / total if total > 0 else 1.0
        )
    return mean.astype(np.float32), basis, energy


def decode(hist: np.ndarray, mean: np.ndarray, basis: np.ndarray,
           code_dtype):
    """Project histories onto the per-node rank-r basis and decode back.

    ``hist`` is (sample, time, node); mean (time, node); basis
    (node, time, rank).  ``code_dtype='float16'`` rounds the codes to fp16
    before decoding to emulate the deployed code storage dtype.
    """
    centered = hist - mean[None]
    code = np.einsum("stn,ntr->srn", centered, basis, optimize=True)
    if code_dtype == "float16":
        code = code.astype(np.float16).astype(np.float32)
    return mean[None] + np.einsum("srn,ntr->stn", code, basis, optimize=True)


def compare_block(raw_idx, raw_dist_sorted, comp_idx, distance,
                  raw_pos, comp_pos):
    """Vectorized identity comparison for one TOD block.

    ``raw_idx``/``comp_idx``: (q, K, n) global endpoints with -1 where
    unavailable.  ``raw_pos``/``comp_pos``: matching block-local candidate
    positions for distance lookups.  ``raw_dist_sorted``: sorted raw
    distances (q, K, n).  ``distance``: full raw distance matrix
    (q, cand, n).
    """
    raw_ok = raw_idx >= 0
    comp_ok = comp_idx >= 0
    top1_valid = raw_ok[:, 0, :] & comp_ok[:, 0, :]
    top1_same = (raw_idx[:, 0, :] == comp_idx[:, 0, :]) & top1_valid
    full_valid = raw_ok.all(axis=1) & comp_ok.all(axis=1)

    # Membership of each raw slot in the compressed top-8 set.
    eq = raw_idx[:, :, None, :] == comp_idx[:, None, :, :]  # (q, K, K, n)
    in_comp = eq.any(axis=2) & full_valid[:, None, :]      # (q, K, n)
    overlap = in_comp.sum(axis=1)                          # (q, n)
    overlap_sum = int(overlap[full_valid].sum())
    slots_full = int(full_valid.sum())

    raw_zero = full_valid[:, None, :] & (raw_dist_sorted == 0.0)
    zero_total = raw_zero.sum(axis=1)                      # (q, n)
    zero_kept = (raw_zero & in_comp).sum(axis=1)
    has_zero = zero_total > 0
    zerodist_slots = int(has_zero.sum())
    zerodist_any = int((zero_kept[has_zero] > 0).sum())
    zerodist_set_sum = float(
        (zero_kept[has_zero] / zero_total[has_zero]).sum()
    ) if zerodist_slots else 0.0

    d1 = raw_dist_sorted[:, 0, :]
    d2 = raw_dist_sorted[:, 1, :]
    both_finite = np.isfinite(d1) & np.isfinite(d2)
    tied = both_finite & (d1 == d2)

    changed = top1_valid & ~top1_same
    changed_count = int(changed.sum())
    equal_swaps = 0
    excess = np.empty(0, dtype=np.float64)
    excess_inf = 0
    if changed_count:
        rows, nodes = np.nonzero(changed)
        old_positions = np.where(
            raw_pos[rows, 0, nodes] >= 0, raw_pos[rows, 0, nodes], 0
        )
        new_positions = np.where(
            comp_pos[rows, 0, nodes] >= 0, comp_pos[rows, 0, nodes], 0
        )
        d_old = distance[rows, old_positions, nodes]
        d_new = distance[rows, new_positions, nodes]
        equal_swaps = int(np.sum(d_new == d_old))
        excess_inf = int(np.sum(~np.isfinite(d_new)))
        diff = (d_new - d_old)[d_new != d_old].astype(np.float64)
        excess = diff[np.isfinite(diff)]

    d1_finite = d1[np.isfinite(d1)]

    return {
        "slots_top1": int(top1_valid.sum()),
        "top1_same": int(top1_same.sum()),
        "slots_full": slots_full,
        "overlap_sum": overlap_sum,
        "tie_slots": int(tied.sum()),
        "tie_both_finite": int(both_finite.sum()),
        "untied_top1_valid": int((top1_valid & ~tied).sum()),
        "untied_top1_same": int((top1_same & ~tied).sum()),
        "zerodist_slots": zerodist_slots,
        "zerodist_any": zerodist_any,
        "zerodist_set_sum": zerodist_set_sum,
        "changed_count": changed_count,
        "equal_swaps": equal_swaps,
        "excess_inf": excess_inf,
        "d1_median": float(np.median(d1_finite)) if d1_finite.size else None,
        "d1_mean": float(d1_finite.mean()) if d1_finite.size else None,
        "excess": excess,
    }


def main():
    print("Loading data and caches ...", flush=True)
    raw_data, _ = load_raw_data(file_path_dict[DATASET])
    data = np.asarray(raw_data[..., 0], dtype=np.float32)  # (T, N)
    with np.load(ROOT / "datasets" / DATASET / f"{DATASET}_time.npz") as z:
        time_data = np.asarray(z["time"])
    with np.load(
        ROOT / "datasets" / DATASET
        / f"{DATASET}_12to12_index_scaler.npz", allow_pickle=False
    ) as z:
        idx_train = np.asarray(z["idx_train"], dtype=np.int64)
        idx_val = np.asarray(z["idx_val"], dtype=np.int64)
    with np.load(CANDIDATE_CACHE, allow_pickle=False) as z:
        cached_val = np.asarray(z["candidate_index_val"], dtype=np.int64)
        for key, expected in (
            ("k", K), ("input_length", INPUT_LENGTH),
            ("min_valid_history", MIN_VALID_HISTORY),
        ):
            actual = int(np.asarray(z[key]).item())
            if actual != expected:
                raise ValueError(f"candidate cache {key}={actual}, expected {expected}")
        rule = str(np.asarray(z["retrieval_rule"]).item())
        order = str(np.asarray(z["candidate_order"]).item())
        if rule != "masked_history_mse" or not order.endswith("ascending"):
            raise ValueError(f"unexpected cache rule/order: {rule}/{order}")
        cache_idx_val = np.asarray(z["idx_val"], dtype=np.int64)
    if not np.array_equal(cache_idx_val, idx_val):
        raise ValueError("candidate cache idx_val disagrees with index scaler")

    x_offsets = np.arange(-(INPUT_LENGTH - 1), 1, dtype=np.int64)
    query_tod = time_data[idx_val, 0].astype(np.int64)
    train_tod = time_data[idx_train, 0].astype(np.int64)
    num_nodes = data.shape[1]

    # ---------------- Correctness gate: reproduce deployed selection -------
    print("Gate: reproducing deployed raw-distance selection on val ...",
          flush=True)
    reproduced = np.full((len(idx_val), K, num_nodes), -1, dtype=np.int64)
    blocks = {}
    for tod in np.unique(query_tod):
        q_pos = np.flatnonzero(query_tod == tod)
        q_group = idx_val[q_pos]
        c_group = idx_train[train_tod == tod]
        q_hist = data[q_group[:, None] + x_offsets[None], :, ].astype(
            np.float32, copy=False)
        c_hist = data[c_group[:, None] + x_offsets[None], :, ].astype(
            np.float32, copy=False)
        distance = masked_history_distance(
            q_hist, c_hist, min_valid_history=MIN_VALID_HISTORY,
            metric="masked_mse",
        )
        nearest, nearest_distance = select_topk(distance, K)
        selected = c_group[nearest]
        finite = np.isfinite(nearest_distance)
        selected = np.where(finite, selected, -1)
        reproduced[q_pos] = selected
        blocks[int(tod)] = (q_pos, q_group, c_group, q_hist, c_hist,
                            distance, nearest, nearest_distance)
    if not np.array_equal(reproduced, cached_val):
        mismatch = int(np.sum(reproduced != cached_val))
        raise RuntimeError(
            f"Gate FAILED: reproduced selection differs from deployed cache "
            f"in {mismatch} entries; aborting without reporting results."
        )
    print("Gate PASSED: bit-exact reproduction of deployed candidate indices.",
          flush=True)

    all_d1 = np.concatenate([
        block[7][:, 0, :].ravel() for block in blocks.values()
    ])
    all_d1 = all_d1[np.isfinite(all_d1)]
    global_d1_median = float(np.median(all_d1))
    global_d1_mean = float(all_d1.mean())
    print(f"Raw top-1 distance scale: median={global_d1_median:.2f} "
          f"mean={global_d1_mean:.2f}", flush=True)

    # ---------------- PCA on train histories -------------------------------
    print("Fitting per-node PCA on train histories ...", flush=True)
    train_hist = data[idx_train[:, None] + x_offsets[None], :, ].astype(
        np.float32, copy=False)
    pcas = {}
    for rank in RANKS:
        mean, basis, energy = fit_history_pca(train_hist, rank)
        pcas[rank] = (mean, basis, energy)
        print(f"  rank {rank:2d}: mean explained energy = "
              f"{float(energy.mean()):.6f}", flush=True)
    with np.load(FUTURES_PCA_CACHE, allow_pickle=False) as z:
        futures_energy = np.asarray(z["retrieval_temporal_explained_ratio"],
                                    dtype=np.float64)
    print(f"  futures PCA r8 explained energy (deployed cache) = "
          f"{float(futures_energy.mean()):.6f}", flush=True)

    # ---------------- Compressed-domain selections -------------------------
    # ``keep_mask`` variants zero out decoded entries wherever the raw history
    # was missing, emulating a deployment that stores the 12-bit validity
    # bitmap alongside the codes.  This isolates pure rank-truncation error
    # from the mean-imputation artifact (which can newly admit candidates
    # whose raw history had fewer than min_valid_history valid points).
    variants = [("mem_r%d" % r, r, None, False) for r in RANKS]
    variants.append(("mem_r8_fp16", 8, "float16", False))
    variants.append(("both_r8", 8, None, False))
    variants.append(("mem_r12_masked", 12, None, True))
    variants.append(("mem_r8_masked", 8, None, True))

    results = {}
    for name, rank, code_dtype, keep_mask in variants:
        print(f"Variant {name}: re-selecting candidates ...", flush=True)
        mean, basis, _ = pcas[rank]
        totals = None
        for tod, (q_pos, q_group, c_group, q_hist, c_hist,
                  distance, nearest, nearest_distance) in blocks.items():
            c_dec = decode(c_hist, mean, basis, code_dtype)
            if keep_mask:
                c_dec = np.where(c_hist != 0, c_dec, 0.0)
            if name.startswith("both"):
                q_dec = decode(q_hist, mean, basis, code_dtype)
                q_used = (np.where(q_hist != 0, q_dec, 0.0)
                          if keep_mask else q_dec)
            else:
                q_used = q_hist
            comp_distance = masked_history_distance(
                q_used, c_dec, min_valid_history=MIN_VALID_HISTORY,
                metric="masked_mse",
            )
            comp_nearest, comp_sorted = select_topk(comp_distance, K)
            selected = c_group[comp_nearest]
            finite = np.isfinite(comp_sorted)
            selected = np.where(finite, selected, -1)

            block = compare_block(
                reproduced[q_pos], nearest_distance, selected, distance,
                nearest, comp_nearest,
            )
            if totals is None:
                totals = {k: v for k, v in block.items()}
                totals["excess"] = block["excess"].copy()
            else:
                totals["excess"] = np.concatenate([totals["excess"],
                                                   block["excess"]])
                for key in (
                    "slots_top1", "top1_same", "slots_full", "overlap_sum",
                    "tie_slots", "tie_both_finite", "untied_top1_valid",
                    "untied_top1_same", "zerodist_slots", "zerodist_any",
                    "zerodist_set_sum", "changed_count", "equal_swaps",
                    "excess_inf",
                ):
                    totals[key] += block[key]

        excess = totals["excess"]
        summary = {
            "top1_agreement": totals["top1_same"] / max(totals["slots_top1"], 1),
            "top8_overlap": totals["overlap_sum"] / max(totals["slots_full"] * K, 1),
            "tie_fraction": totals["tie_slots"] / max(totals["tie_both_finite"], 1),
            "top1_agreement_untied": (
                totals["untied_top1_same"] / max(totals["untied_top1_valid"], 1)
            ),
            "zerodist_any_retention": (
                totals["zerodist_any"] / max(totals["zerodist_slots"], 1)
            ),
            "zerodist_set_retention": (
                totals["zerodist_set_sum"] / max(totals["zerodist_slots"], 1)
            ),
            "changed_top1_slots": totals["changed_count"],
            "equal_distance_swaps": totals["equal_swaps"],
            "replacement_slots_with_inf_raw_distance": totals["excess_inf"],
            "raw_top1_distance_median": global_d1_median,
            "raw_top1_distance_mean": global_d1_mean,
            "replacement_excess_median": (
                float(np.median(excess)) if excess.size else 0.0
            ),
            "replacement_excess_mean": (
                float(excess.mean()) if excess.size else 0.0
            ),
            "slots_top1": totals["slots_top1"],
            "slots_full": totals["slots_full"],
            "zerodist_slots": totals["zerodist_slots"],
        }
        results[name] = summary
        print(f"  top1={summary['top1_agreement']:.4f} "
              f"top8_overlap={summary['top8_overlap']:.4f} "
              f"tie_frac={summary['tie_fraction']:.4f} "
              f"untied_top1={summary['top1_agreement_untied']:.4f} "
              f"zerodist_any={summary['zerodist_any_retention']:.4f}",
              flush=True)

    payload = {
        "dataset": DATASET,
        "split": "val",
        "protocol": {
            "metric": "masked_mse",
            "min_valid_history": MIN_VALID_HISTORY,
            "k": K,
            "pca_fit_split": "train",
            "primary_variant": "mem_r8",
            "gate": "bit-exact reproduction of deployed candidate_index_val",
        },
        "history_pca_explained_energy": {
            str(r): float(pcas[r][2].mean()) for r in RANKS
        },
        "futures_pca_explained_energy_r8_deployed": float(futures_energy.mean()),
        "variants": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
