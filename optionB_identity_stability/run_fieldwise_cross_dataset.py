"""Cross-dataset validation of the field-wise compression law.

PEMS04 result to be validated or refuted (from run_aip_tc_sparse.py):
  * exact fires on 9.48% of slots, yet carries a +0.329 MAE gain;
  * low-rank r8 on the exact VALUES costs +0.122 (prohibitive);
  * sparsity (store E only where it fires) is exactly lossless, -87.4% on E;
  * sparsity + fp16 is -92.1% on E at +0.0009.

This script repeats the same four measurements on PEMS03/04/07/08 under one
single protocol, using the UNCOMPRESSED caches (no dataset has a latc_pca_r8
cache except PEMS04), so all four share an identical reference point:

    C0 : T = retrieval_val (raw), E = exact_retrieval_val (raw fp32)

Pre-registered expectations (stated before looking at the numbers):
  E1  sparsity is lossless everywhere (|delta| < 1e-6 in MAE);
  E2  low-rank r8 on E costs >= 0.05 on every dataset (prohibitive everywhere);
  E3  the exact channel contributes > 0.05 MAE on every dataset
      (otherwise the sparsity saving buys nothing worth protecting).

E1/E2/E3 all holding => the field-wise law generalises.
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

OUT = Path(__file__).resolve().parent / "fieldwise_cross_dataset_val.json"

HORIZON = 12
RANK = 8
Y_OFF = np.arange(1, 13, dtype=np.int64)

DATASETS = {
    "PEMS03": "TLAST_improved_PEMS03_12to12_exppaper_pems03_s1_final_joint",
    "PEMS04": "TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3",
    "PEMS07": "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint",
    "PEMS08": "TLAST_improved_PEMS08_12to12_exppaper_pems08_s1_final_joint",
}


def fit_pca_cov(train_future: np.ndarray, rank: int, min_valid: int,
                block: int = 64):
    """Per-node PCA via covariance eigendecomposition (cheaper than SVD).

    train_future: (S, H, N).  Returns mean (H,N), basis (N,H,rank),
    explained energy (N,).  Nodes are processed in blocks so that PEMS07
    (883 nodes) never materialises a full (S, H, N) centred copy.
    """
    n_samp, n_time, n_node = train_future.shape
    mean_f = np.empty((n_time, n_node), dtype=np.float32)
    basis = np.empty((n_node, n_time, rank), dtype=np.float32)
    energy = np.empty((n_node,), dtype=np.float64)
    for start in range(0, n_node, block):
        end = min(start + block, n_node)
        sub = np.ascontiguousarray(train_future[:, :, start:end])
        observed = sub != 0
        counts = observed.sum(axis=0).astype(np.float64)
        mean_b = np.zeros((n_time, end - start), dtype=np.float64)
        np.divide(np.where(observed, sub, 0.0).sum(axis=0), counts,
                  out=mean_b, where=counts > 0)
        mean_b = mean_b.astype(np.float32)
        centered = (np.where(observed, sub,
                             np.broadcast_to(mean_b, sub.shape)) - mean_b)
        # ok is (S, b): per-sample validity floor, broadcast over the horizon
        ok = (observed.sum(axis=1) >= min_valid)
        centered = centered * ok[:, None, :].astype(np.float32)
        # explicit matmul; einsum's optimiser picks an (S,S,H,b) path here
        x = np.ascontiguousarray(np.transpose(centered, (2, 0, 1)))  # (b,S,H)
        cov = np.matmul(x.transpose(0, 2, 1), x).astype(np.float64)   # (b,H,H)
        del x
        mean_f[:, start:end] = mean_b
        for j in range(end - start):
            vals, vecs = np.linalg.eigh(cov[j])
            order = np.argsort(vals)[::-1]
            vals = np.clip(vals[order], 0.0, None)
            vecs = vecs[:, order]
            basis[start + j] = vecs[:, :rank].astype(np.float32)
            total = float(vals.sum())
            energy[start + j] = (float(vals[:rank].sum()) / total
                                 if total > 0 else 1.0)
        del sub, observed, centered, cov
    return mean_f, basis, energy


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
    q0_c, q1_c, q2_c = quality_scalars(cache, scaler_std, HORIZON)

    p_val = np.asarray(np.load(backbone_path)["backbone_pred_val"],
                       dtype=np.float32)
    y_raw = raw[idx_val[:, None] + Y_OFF[None], :]
    y_z = ((y_raw - scaler_mean) / scaler_std).astype(np.float32)
    valid_q = y_raw != 0

    pipe = FrozenPipeline(exp_dir, HORIZON, num_nodes)
    pipe.set_codec(None, None, quantize=False)
    w = pipe.w

    def mae(pred_z):
        return (float((np.abs(pred_z[..., 0] - y_z) * valid_q).sum()
                      / valid_q.sum()) * scaler_std)

    def run(T, E, av=None):
        return mae(pipe.forward(p_val, T, E,
                                avail if av is None else av,
                                q0_c, q1_c, q2_c))

    fired = avail > 0.5
    fire_rate = float(fired.mean())

    base = run(T_raw, e_val)

    # sparsity: unfired slots provably never read E
    mean_fill = np.zeros((HORIZON, num_nodes), dtype=np.float32)
    e_sparse = np.where(fired, e_val,
                        np.broadcast_to(mean_fill[None, ..., None],
                                        e_val.shape))
    mae_sparse = run(T_raw, e_sparse)
    e_sparse_fp16 = np.where(
        fired, e_val.astype(np.float16).astype(np.float32),
        np.broadcast_to(mean_fill[None, ..., None], e_val.shape))
    mae_sparse_fp16 = run(T_raw, e_sparse_fp16)

    # low rank on the exact values
    train_future = raw[idx_train[:, None] + Y_OFF[None], :].astype(
        np.float32, copy=False)
    fut_mean_raw, fut_basis, fut_energy = fit_pca_cov(train_future, RANK, 6)
    fut_mean_z = ((fut_mean_raw - scaler_mean) / scaler_std).astype(np.float32)

    def codec_field(values, mean_z, basis):
        centered = values[..., 0] - mean_z[None]
        code = np.einsum("shn,nhr->srn", centered, basis, optimize=True)
        return (mean_z[None] + np.einsum("srn,nhr->shn",
                                         code.astype(np.float32), basis,
                                         optimize=True))[..., None]

    mae_e_r8 = run(T_raw, codec_field(e_val, fut_mean_z, fut_basis))
    # low rank on the mean field T (the deployed-asymmetric counterpart)
    mae_t_r8 = run(codec_field(T_raw, fut_mean_z, fut_basis), e_val)
    # both fields
    mae_both = run(codec_field(T_raw, fut_mean_z, fut_basis),
                   codec_field(e_val, fut_mean_z, fut_basis))

    # exact channel contribution
    zero_avail = np.zeros_like(avail)
    mae_no_exact = run(T_raw, e_val, av=zero_avail)
    exact_gain = mae_no_exact - base

    # recon error over fired slots, for the self-consistency check
    e_r8 = codec_field(e_val, fut_mean_z, fut_basis)
    dil = (avail[..., 0] > 0.5) & valid_q
    recon_z = float(np.abs(e_r8 - e_val)[..., 0][dil].mean())
    expected_cost = float(np.abs(w).mean()) * fire_rate * recon_z * scaler_std

    result = {
        "nodes": int(num_nodes),
        "val_samples": int(len(idx_val)),
        "fire_rate": fire_rate,
        "exact_gate_mean_abs_w": float(np.abs(w).mean()),
        "baseline_C0": base,
        "arms": {
            "C0_raw": 0.0,
            "S_sparse_f32": mae_sparse - base,
            "SQ_sparse_fp16": mae_sparse_fp16 - base,
            "E_lowrank_r8": mae_e_r8 - base,
            "T_lowrank_r8": mae_t_r8 - base,
            "both_lowrank_r8": mae_both - base,
            "exact_channel_off": mae_no_exact - base,
        },
        "exact_channel_gain": exact_gain,
        "gain_per_fired_slot": exact_gain / fire_rate if fire_rate else None,
        "futures_pca_energy_rank8": float(fut_energy.mean()),
        "self_consistency": {
            "recon_mae_z_fired": recon_z,
            "naive_expected_cost": expected_cost,
            "measured_cost": mae_e_r8 - base,
        },
        "E_bits_per_slot": {
            "dense_f32": 32.0,
            "sparse_f32_plus_bitmap": 32.0 * fire_rate + 1.0,
            "sparse_f16_plus_bitmap": 16.0 * fire_rate + 1.0,
        },
        "E_saving_vs_dense_f32": {
            "sparse_f32": 1.0 - (32.0 * fire_rate + 1.0) / 32.0,
            "sparse_f16": 1.0 - (16.0 * fire_rate + 1.0) / 32.0,
        },
    }
    print(f"\n=== {ds} (N={num_nodes}, val={len(idx_val)}) ===", flush=True)
    print(f"  fire_rate            {fire_rate:.4f}")
    print(f"  C0 baseline MAE      {base:.7f}")
    for k, v in result["arms"].items():
        print(f"  {k:22s} {v:+.7f}")
    print(f"  exact channel gain   {exact_gain:+.7f} "
          f"(per fired slot {result['gain_per_fired_slot']:.4f})")
    print(f"  E saving sparse_f16  "
          f"{result['E_saving_vs_dense_f32']['sparse_f16'] * 100:.2f}%")
    print(f"  self-consistency     expected {expected_cost:.5f} vs "
          f"measured {mae_e_r8 - base:+.5f}")
    return result


def main():
    results = {}
    for ds, exp_name in DATASETS.items():
        try:
            results[ds] = run_dataset(ds, exp_name)
        except Exception as exc:  # keep going; report the failure honestly
            print(f"\n=== {ds} FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[ds] = {"error": f"{type(exc).__name__}: {exc}"}

    print("\n================ SUMMARY ================")
    hdr = ("dataset  fire    C0        S(sparse)  SQ(fp16)   "
           "E_r8      T_r8      exact_gain")
    print(hdr)
    for ds, r in results.items():
        if "error" in r:
            print(f"{ds:8s} FAILED")
            continue
        a = r["arms"]
        print(f"{ds:8s} {r['fire_rate']:.4f}  {r['baseline_C0']:.5f}  "
              f"{a['S_sparse_f32']:+.7f} {a['SQ_sparse_fp16']:+.7f} "
              f"{a['E_lowrank_r8']:+.5f} {a['T_lowrank_r8']:+.5f} "
              f"{r['exact_channel_gain']:+.5f}")

    ok = [r for r in results.values() if "error" not in r]
    verdict = {
        "E1_sparsity_lossless_everywhere": bool(
            ok and all(abs(r["arms"]["S_sparse_f32"]) < 1e-6 for r in ok)),
        "E2_lowrank_prohibitive_everywhere": bool(
            ok and all(r["arms"]["E_lowrank_r8"] >= 0.05 for r in ok)),
        "E3_exact_channel_matters_everywhere": bool(
            ok and all(r["exact_channel_gain"] > 0.05 for r in ok)),
        "datasets_completed": [d for d, r in results.items()
                               if "error" not in r],
        "datasets_failed": [d for d, r in results.items() if "error" in r],
    }
    verdict["fieldwise_law_generalises"] = bool(
        verdict["E1_sparsity_lossless_everywhere"]
        and verdict["E2_lowrank_prohibitive_everywhere"]
        and verdict["E3_exact_channel_matters_everywhere"])
    print("\n" + json.dumps(verdict, indent=2))

    OUT.write_text(json.dumps({"results": results, "verdict": verdict},
                              indent=2), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
