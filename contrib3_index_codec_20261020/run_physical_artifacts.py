"""Materialise the FCL artifacts on disk and re-account on *persisted* bytes.

Why this script exists
----------------------
The 2026-10-20 accounting was computed on **array bytes** (``ndarray.nbytes``).
The shipped caches are ``np.savez_compressed`` files, and deflate already
removes 2.3-3.8x from them.  A 6-bit packed index, by contrast, is nearly
incompressible.  So "array bytes -81%" and "file bytes -29.5%" are both true
and are NOT interchangeable.  This script therefore:

  1. writes three real artifacts per dataset (all_f32 / deployed / fcl) with
     ``np.savez_compressed`` and measures the file size on disk,
  2. reads them back and decodes, asserting the decoded arrays are bit-identical
     to the quantised arms used in the in-memory runs,
  3. recomputes validation MAE end-to-end **from the decoded artifact**, so the
     accuracy claim no longer rests on an in-memory simulation,
  4. measures decode wall time (single run, not a latency benchmark).

Artifacts per split are written, measured and then deleted unless ``--keep`` is
given, so no multi-GB residue is left behind.  The FCL + 6-bit-index artifact of
each dataset is kept (it is small).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emulate_and_build_residual_oracle import (                     # noqa: E402
    FrozenPipeline, quality_scalars,
)
from codec_fields import (                                          # noqa: E402
    encode_bitmap, decode_bitmap, encode_e_sparse, decode_e_sparse,
    encode_fd, decode_fd, encode_hd, decode_hd, encode_t, decode_t,
    fit_t_basis, RANK,
)
from index_codec import TodIndexCodec                               # noqa: E402

HORIZON = 12
SPLITS = ("train", "val", "test")
HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "physical"
OUT_JSON = HERE / "physical_artifacts.json"

DATASETS = {
    "PEMS03": "TLAST_improved_PEMS03_12to12_exppaper_pems03_s1_final_joint",
    # PEMS04 ships the latc_pca_r8 joint3 checkpoint (name differs on purpose)
    "PEMS04": "TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3",
    "PEMS07": "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint",
    "PEMS08": "TLAST_improved_PEMS08_12to12_exppaper_pems08_s1_final_joint",
}


def quant_fp16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def quant_int16_per_node(x):
    """Reference quantiser, identical to run_fieldwise_global_accounting.py."""
    x = np.asarray(x, dtype=np.float32)
    absmax = np.abs(x).max(axis=0)
    scale = np.where(absmax > 0, absmax / 32767.0, 1.0).astype(np.float32)
    q = np.clip(np.round(x / scale[None]), -32767, 32767).astype(np.int16)
    return (q.astype(np.float32) * scale[None]).astype(np.float32)


def write_measure(arrays, path):
    t0 = time.perf_counter()
    np.savez_compressed(path, **arrays)
    elapsed = time.perf_counter() - t0
    return os.path.getsize(path), elapsed


def plain(x):
    return np.ascontiguousarray(x)


def run_dataset(ds, exp_name, keep):
    ds_dir = ROOT / "datasets" / ds
    exp_dir = ROOT / "EXP_experiments/TLAST_improved" / ds / exp_name
    cache_path = ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz"
    index_path = ds_dir / f"{ds}_12to12_analog_k8_candidate_indices.npz"
    backbone_path = ds_dir / f"{ds}_12to12_backbone_predictions_joint3.npz"

    meta = np.load(ds_dir / f"{ds}_12to12_index_scaler.npz")
    idx_train = np.asarray(meta["idx_train"], dtype=np.int64)
    idx_val = np.asarray(meta["idx_val"], dtype=np.int64)
    scaler_mean = float(np.asarray(meta["scaler_mean"]).item())
    scaler_std = float(np.asarray(meta["scaler_std"]).item())

    from utils.datasets_utils import file_path_dict, load_raw_data
    raw_all, _ = load_raw_data(file_path_dict[ds])
    raw = np.asarray(raw_all[..., 0], dtype=np.float32)
    num_nodes = raw.shape[1]

    time_data = np.load(ds_dir / f"{ds}_time.npz")["time"]
    tod_codec = TodIndexCodec(time_data, idx_train)

    cache = np.load(cache_path, allow_pickle=False)
    print(f"\n=== {ds}: fitting T basis (rank {RANK}) on retrieval_train ...",
          flush=True)
    t0 = time.perf_counter()
    basis_mean, basis = fit_t_basis(plain(cache["retrieval_train"][..., 0]))
    fit_s = time.perf_counter() - t0

    pipe = FrozenPipeline(exp_dir, HORIZON, num_nodes)
    pipe.set_codec(None, None, quantize=False)
    p_val = np.asarray(np.load(backbone_path)["backbone_pred_val"],
                       dtype=np.float32)
    index_cache = np.load(index_path, allow_pickle=False)

    per_split, verify, timings = {}, {}, {}
    for split in SPLITS:
        print(f"--- {split}", flush=True)
        t = plain(cache[f"retrieval_{split}"][..., 0])
        e = plain(cache[f"exact_retrieval_{split}"][..., 0])
        avail = plain(cache[f"exact_available_{split}"])   # (S,H,N,1) kept
        fd = plain(cache[f"future_dispersion_{split}"][..., 0])
        hd = plain(cache[f"history_distance_{split}"])
        fired = avail[..., 0] > 0.5

        code = encode_t(t, basis_mean, basis)
        e_vals = encode_e_sparse(e, fired)
        bitmap = encode_bitmap(fired)
        fd_code = encode_fd(fd)
        hd_q, hd_scale = encode_hd(hd, bits=16)

        # ---- config: all float32 (reference) ------------------------------
        a_all = {"T": t, "E": e, "fd": fd, "hd": hd, "B": avail.astype(np.uint8)}
        # ---- config: deployed form (T = r8 fp16 code, rest f32) -----------
        a_dep = {"T_code": code, "T_mean": basis_mean, "T_basis": basis,
                 "E": e, "fd": fd, "hd": hd, "B": avail.astype(np.uint8)}
        # ---- config: FCL --------------------------------------------------
        a_fcl = {"T_code": code, "T_mean": basis_mean, "T_basis": basis,
                 "E_values": e_vals, "bitmap": bitmap, "fd_code": fd_code,
                 "hd_code": hd_q, "hd_scale": hd_scale,
                 "n_slots": np.int64(fired.size), "horizon": np.int64(HORIZON),
                 "num_nodes": np.int64(num_nodes), "rank": np.int64(RANK)}

        sizes = {}
        scratch = OUTDIR / "_scratch.npz"        # overwritten, never deleted
        for name, arrays in (("all_f32", a_all), ("deployed", a_dep),
                             ("fcl", a_fcl)):
            size, secs = write_measure(arrays, scratch)
            sizes[name] = size
            print(f"    {name:9s} {size/1e6:9.2f} MB  (write {secs:5.1f}s)",
                  flush=True)

        # ---- read the fcl artifact back and verify bit-exactness ----------
        with np.load(scratch, allow_pickle=False) as z:
            mask = decode_bitmap(np.asarray(z["bitmap"]), fired.size)
            dec_t0 = time.perf_counter()
            t_dec = decode_t(np.asarray(z["T_code"]), np.asarray(z["T_mean"]),
                             np.asarray(z["T_basis"]))
            dec_t1 = time.perf_counter()
            e_dec = decode_e_sparse(np.asarray(z["E_values"]), fired, e.shape)
            dec_t2 = time.perf_counter()
            fd_dec = decode_fd(np.asarray(z["fd_code"]))
            dec_t3 = time.perf_counter()
            hd_dec = decode_hd(np.asarray(z["hd_code"]), np.asarray(z["hd_scale"]))
            dec_t4 = time.perf_counter()
        decode_total = dec_t4 - dec_t0

        # ---- 6-bit index (separate member; npz compresses members
        # independently, so its bytes add to the FCL payload exactly) ------
        idx_arr = np.asarray(index_cache[f"candidate_index_{split}"],
                             dtype=np.int32)
        t1 = time.perf_counter()
        code_idx = tod_codec.encode(idx_arr)
        packed_idx = tod_codec.pack(code_idx)
        idx_pack_s = time.perf_counter() - t1
        idx_size, _ = write_measure({"idx_code": packed_idx}, scratch)
        sizes["index6"] = idx_size
        print(f"    index6    {idx_size/1e6:9.2f} MB  "
              f"(raw {idx_arr.nbytes/1e6:8.2f} MB, pack {idx_pack_s:4.1f}s)",
              flush=True)
        del idx_arr, code_idx, packed_idx

        if keep and ds == "PEMS04" and split == "val":
            write_measure(a_fcl, OUTDIR / "PEMS04_val_fcl_artifact.npz")

        e_expect = np.where(fired, quant_fp16(e), np.float32(0.0))
        fd_expect = quant_fp16(fd)
        hd_expect = quant_int16_per_node(hd)
        verify[split] = {
            "mask_bit_identical": bool(np.array_equal(mask, fired.reshape(-1))),
            "E_bit_identical": bool(np.array_equal(e_dec, e_expect)),
            "fd_bit_identical": bool(np.array_equal(fd_dec, fd_expect)),
            "hd_bit_identical": bool(np.array_equal(hd_dec, hd_expect)),
            "T_max_abs_error_vs_raw": float(np.abs(t_dec - t).max()),
            "decode_ms": {
                "T": 1e3 * (dec_t1 - dec_t0), "E": 1e3 * (dec_t2 - dec_t1),
                "fd": 1e3 * (dec_t3 - dec_t2), "hd": 1e3 * (dec_t4 - dec_t3),
                "all": 1e3 * decode_total,
            },
        }
        print("    verify: " + ", ".join(
            f"{k}={v}" for k, v in verify[split].items() if k != "decode_ms")
            + f" | decode {1e3*decode_total:.0f} ms", flush=True)

        # ---- end-to-end MAE on val, straight from the artifact -----------
        if split == "val":
            y_raw = raw[idx_val[:, None] + np.arange(1, HORIZON + 1)[None], :]
            y_z = ((y_raw - scaler_mean) / scaler_std).astype(np.float32)
            valid_q = y_raw != 0

            def mae(pred_z):
                return (float((np.abs(pred_z[..., 0] - y_z) * valid_q).sum()
                              / valid_q.sum()) * scaler_std)

            def forward(T_arr, E_arr, fd_arr, hd_arr):
                q0, q1, q2 = quality_scalars(
                    {"history_distance_val": hd_arr,
                     "future_dispersion_val": fd_arr[..., None],
                     "exact_available_val": avail},
                    scaler_std, HORIZON)
                return mae(pipe.forward(p_val, T_arr[..., None],
                                        E_arr[..., None], avail, q0, q1, q2))

            timings["val_C0"] = forward(t, e, fd, hd)
            timings["val_fcl"] = forward(t_dec, e_dec, fd_dec, hd_dec)
            ver = verify[split]
            ver["val_MAE_C0_raw"] = timings["val_C0"]
            ver["val_MAE_from_artifact"] = timings["val_fcl"]
            ver["val_MAE_delta"] = timings["val_fcl"] - timings["val_C0"]
            print(f"    MAE: C0(raw) {timings['val_C0']:.5f}  "
                  f"artifact {timings['val_fcl']:.5f}  "
                  f"delta {ver['val_MAE_delta']:+.5f}", flush=True)

        per_split[split] = sizes
        del t, e, avail, fd, hd, code, e_vals, bitmap, fd_code, hd_q, hd_scale
        del t_dec, e_dec, fd_dec, hd_dec, e_expect, fd_expect, hd_expect, mask
        del a_all, a_dep, a_fcl

    cache.close()
    index_cache.close()

    tot = {k: sum(per_split[s][k] for s in SPLITS)
           for k in ("all_f32", "deployed", "fcl", "index6")}
    tot["fcl_index6"] = tot["fcl"] + tot["index6"]

    shipped = {
        "cache_file_MB": os.path.getsize(cache_path) / 1e6,
        "index_file_MB": os.path.getsize(index_path) / 1e6,
    }
    latc_path = ds_dir / f"{ds}_12to12_analog_k8_exact_mean_latc_pca_r8_fp16.npz"
    if latc_path.is_file():
        shipped["deployed_cache_file_MB"] = os.path.getsize(latc_path) / 1e6
        shipped["total_deployed_MB"] = (shipped["deployed_cache_file_MB"]
                                        + shipped["index_file_MB"])
    shipped["total_MB"] = shipped["cache_file_MB"] + shipped["index_file_MB"]

    fcl_mb = tot["fcl_index6"] / 1e6
    return {
        "exp_dir": exp_name,
        "num_nodes": num_nodes,
        "T_basis_fit_s": fit_s,
        "per_split_file_MB": {s: {k: v / 1e6 for k, v in per_split[s].items()}
                              for s in SPLITS},
        "total_file_MB": {k: v / 1e6 for k, v in tot.items()},
        "shipped_reference_MB": shipped,
        "saving_vs_all_f32": 1.0 - fcl_mb / (tot["all_f32"] / 1e6),
        "saving_vs_deployed_form": 1.0 - fcl_mb / (tot["deployed"] / 1e6),
        "saving_vs_shipped_files": 1.0 - fcl_mb / shipped["total_MB"],
        "saving_vs_shipped_deployed": (
            1.0 - fcl_mb / shipped["total_deployed_MB"]
            if "total_deployed_MB" in shipped else None),
        "verify": verify,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--keep", action="store_true",
                    help="keep the FCL artifacts (default: only PEMS04 kept)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for ds in args.datasets:
        try:
            results[ds] = run_dataset(ds, DATASETS[ds], args.keep)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[ds] = {"error": f"{type(exc).__name__}: {exc}"}
        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n================ DISK-BYTES SUMMARY (MB, real files) ================")
    print("%-8s %9s %9s %9s %9s %9s %11s %10s"
          % ("ds", "all_f32", "T-r8", "fcl", "fcl+idx6", "shipped",
             "vs all_f32", "vs shipped"))
    for ds, r in results.items():
        if "error" in r:
            print(f"{ds:8s} ERROR {r['error'][:60]}")
            continue
        t = r["total_file_MB"]
        print("%-8s %9.1f %9.1f %9.1f %9.1f %9.1f %10.1f%% %9.1f%%"
              % (ds, t["all_f32"], t["deployed"], t["fcl"], t["fcl_index6"],
                 r["shipped_reference_MB"]["total_MB"],
                 100 * r["saving_vs_all_f32"],
                 100 * r["saving_vs_shipped_files"]))
    print("\n  vs the shipped *deployed* cache where one exists (latc_pca_r8):")
    for ds, r in results.items():
        if "error" in r or r["saving_vs_shipped_deployed"] is None:
            continue
        print("    %-8s %.1f%%" % (ds, 100 * r["saving_vs_shipped_deployed"]))
    print(f"\nJSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
