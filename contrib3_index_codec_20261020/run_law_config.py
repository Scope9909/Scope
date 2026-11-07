"""Does the field-wise law actually *select* the right codec per dataset?

The earlier ablation (`run_codec_ablation.py`) only added up per-field byte
counts.  It never proved that the law-selected configuration is (a) really
smaller on disk and (b) still inside the accuracy gate.  This script closes
that gap by materialising the chosen configuration.

Per dataset, on **validation only**:

  1. measure each candidate T codec in isolation  (r8 low-rank vs bare fp16)
  2. measure whether the B bitpack beats plain uint8 *after* the generic codec
  3. pick the configuration from those two readings alone

Then, **once**, on test: replay C0 and the selected configuration and report
the deltas.  Nothing about test influences the choice.

The rule being tested (the claim C3 makes):
    T   -> rank-8 only if its own validation delta <= GATE, else fp16
    B   -> bitpack only if it beats xz on the raw uint8 mask, else leave it
    E/fd/hd -> always on (bit-exact or already cleared the gate)

Usage:  python run_law_config.py [PEMS08 PEMS07 ...] [--gate 0.01]
"""

import argparse
import io
import json
import lzma
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emulate_and_build_residual_oracle import FrozenPipeline   # noqa: E402
import codec_fields as CF                                      # noqa: E402

HORIZON = 12
HERE = Path(__file__).resolve().parent
OUT = HERE / "law_config.json"
SCRATCH = HERE / "physical" / "_law_scratch.npz"


def scratch_path(ds, t_pref, fd_pref):
    """Per-run scratch name: two concurrent runs must not collide."""
    return HERE / "physical" / f"_law_{ds}_{t_pref}_{fd_pref}.npz"

EXP = {
    "PEMS03": "TLAST_improved_PEMS03_12to12_exppaper_pems03_s1_final_joint",
    "PEMS04": "TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3",
    "PEMS07": "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint",
    "PEMS08": "TLAST_improved_PEMS08_12to12_exppaper_pems08_s1_final_joint",
}


def lz(a, preset=6):
    return len(lzma.compress(np.ascontiguousarray(a).tobytes(),
                             format=lzma.FORMAT_XZ, preset=preset))


def zl(arrays):
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    return b.getbuffer().nbytes


def q_scalars(hd, fd, avail, scaler_std):
    """Split-agnostic clone of quality_scalars().

    ``hd`` (S,N); ``fd`` (S,H,N) already squeezed; ``avail`` (S,H,N,1).
    """
    hd = np.asarray(hd, dtype=np.float32) / np.float32(scaler_std ** 2)
    hd = np.repeat(hd[:, None, :], HORIZON, axis=1)
    fd = np.asarray(fd, dtype=np.float32) / np.float32(scaler_std)
    ex = np.asarray(avail, dtype=np.float32)[..., 0]

    def squash(x):
        x = np.log1p(np.clip(x, 0.0, None))
        return x / (1.0 + x)

    return squash(hd), squash(fd), ex


def mae_fn(y_z, valid, scaler_std):
    n = valid.sum()

    def f(pred_z):
        return float((np.abs(pred_z[..., 0] - y_z) * valid).sum() / n) * scaler_std

    return f


def load_split(cache, split):
    return dict(
        T=np.ascontiguousarray(cache[f"retrieval_{split}"][..., 0]),
        E=np.ascontiguousarray(cache[f"exact_retrieval_{split}"][..., 0]),
        fd=np.ascontiguousarray(cache[f"future_dispersion_{split}"][..., 0]),
        hd=np.ascontiguousarray(cache[f"history_distance_{split}"]),
        B=np.ascontiguousarray(cache[f"exact_available_{split}"]),
    )


def evaluate(pipe, P, y_z, valid, scaler_std, arrays, T_override=None):
    """MAE of the frozen pipeline given a set of (possibly degraded) fields."""
    T = arrays["T"] if T_override is None else T_override
    q0, q1, q2 = q_scalars(arrays["hd"], arrays["fd"], arrays["B"], scaler_std)
    pred = pipe.forward(P, T[..., None], arrays["E"][..., None], arrays["B"],
                        q0, q1, q2)
    return mae_fn(y_z, valid, scaler_std)(pred)


def run_dataset(ds, gate, t_pref="auto", fd_pref="f16", hd_pref=16):
    ds_dir = ROOT / "datasets" / ds
    exp_dir = ROOT / "EXP_experiments/TLAST_improved" / ds / EXP[ds]

    meta = np.load(ds_dir / f"{ds}_12to12_index_scaler.npz")
    idx_train = np.asarray(meta["idx_train"], dtype=np.int64)
    idx_val = np.asarray(meta["idx_val"], dtype=np.int64)
    idx_test = np.asarray(meta["idx_test"], dtype=np.int64)
    s_mean = float(np.asarray(meta["scaler_mean"]).item())
    s_std = float(np.asarray(meta["scaler_std"]).item())

    from utils.datasets_utils import file_path_dict, load_raw_data
    raw_all, _ = load_raw_data(file_path_dict[ds])
    raw = np.asarray(raw_all[..., 0], dtype=np.float32)
    n_nodes = raw.shape[1]

    bb = np.load(ds_dir / f"{ds}_12to12_backbone_predictions_joint3.npz")
    pipe = FrozenPipeline(exp_dir, HORIZON, n_nodes)
    pipe.set_codec(None, None, quantize=False)   # codec applied by us, not by forward()

    cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                    allow_pickle=False)
    print(f"\n=== {ds}: fitting T basis (rank {CF.RANK}) ...", flush=True)
    t0 = time.perf_counter()
    mean, basis = CF.fit_t_basis(load_split(cache, "train")["T"])
    fit_s = time.perf_counter() - t0
    print(f"    fit {fit_s:.1f}s", flush=True)

    val = load_split(cache, "val")
    fired_val = val["B"][..., 0] > 0.5

    def gt(idx, P):
        y_raw = raw[idx[:, None] + np.arange(1, HORIZON + 1)[None], :]
        return ((y_raw - s_mean) / s_std).astype(np.float32), y_raw != 0

    yv, vv = gt(idx_val, np.asarray(bb["backbone_pred_val"], dtype=np.float32))
    P_val = np.asarray(bb["backbone_pred_val"], dtype=np.float32)
    C0_val = evaluate(pipe, P_val, yv, vv, s_std, val)

    # ---- step 1: T codec, measured in isolation on validation ------------
    t_r8 = CF.decode_t(CF.encode_t(val["T"], mean, basis), mean, basis)
    t_f16 = np.asarray(val["T"], np.float32).astype(np.float16).astype(np.float32)
    d_r8 = evaluate(pipe, P_val, yv, vv, s_std, val, T_override=t_r8) - C0_val
    d_f16 = evaluate(pipe, P_val, yv, vv, s_std, val, T_override=t_f16) - C0_val
    # Pareto reading (`run_pareto_t_mode.py`): with xz in the loop, fp16 is
    # SMALLER than rank-8 on all four datasets, and its error is 2-3 orders of
    # magnitude lower.  rank-8 is dominated on both axes, so the law retires it.
    if t_pref == "f16":
        t_mode = "f16"
    elif t_pref in ("r8", "i8"):
        t_mode = t_pref
    else:
        t_mode = "r8" if d_r8 <= gate else "f16"
    print(f"    T isolation: r8 {d_r8:+.5f}  f16 {d_f16:+.5f}  "
          f"-> gate {gate} / pref {t_pref} picks {t_mode}", flush=True)

    # ---- step 2: does bitpack beat xz on the raw mask? -------------------
    bits = CF.encode_bitmap(fired_val)
    b_raw = val["B"].astype(np.uint8)
    lz_bits, lz_raw = lz(bits), lz(b_raw)
    b_gain = 1.0 - lz_bits / lz_raw if lz_raw else 0.0
    b_mode = "bits" if b_gain > 0.01 else "raw"
    print(f"    B: lz(bits) {lz_bits/1e6:.3f} vs lz(raw) {lz_raw/1e6:.3f} "
          f"gain {100*b_gain:+.1f}% -> {b_mode}", flush=True)

    # ---- step 3: assemble the selected configuration ---------------------
    def build(split):
        a = load_split(cache, split)
        fired = a["B"][..., 0] > 0.5
        cfg = {
            "E_values": CF.encode_e_sparse(a["E"], fired),
            "bitmap": CF.encode_bitmap(fired),
        }
        if fd_pref == "zero":
            pass                       # field dropped: sensitivity ladder shows
                                       # it costs nothing (PEMS04 improves)
        elif fd_pref == "log8":
            cfg["fd_code"], cfg["fd_top"] = CF.encode_fd_log(a["fd"], 8)
        elif fd_pref == "log6":
            cfg["fd_code"], cfg["fd_top"] = CF.encode_fd_log(a["fd"], 6)
        elif fd_pref == "log4":
            cfg["fd_code"], cfg["fd_top"] = CF.encode_fd_log(a["fd"], 4)
        elif fd_pref == "int8":
            cfg["fd_code"], cfg["fd_top"] = CF.encode_fd_int8(a["fd"])
        else:
            cfg["fd_code"] = CF.encode_fd(a["fd"])
        cfg["hd_code"], cfg["hd_scale"] = CF.encode_hd(a["hd"], bits=hd_pref)
        if t_mode == "r8":
            cfg["T_code"] = CF.encode_t(a["T"], mean, basis)
            cfg["T_mean"], cfg["T_basis"] = mean, basis
        elif t_mode == "i8":
            cfg["T_i8"], cfg["T_i8_scale"] = CF.encode_hd(a["T"], bits=8)
        else:
            cfg["T_f16"] = a["T"].astype(np.float16)
        if b_mode == "raw":
            cfg["B_raw"] = a["B"].astype(np.uint8)
        return cfg, a

    def decode(cfg, ref):
        fired = ref["B"][..., 0] > 0.5
        if "T_code" in cfg:
            T = CF.decode_t(cfg["T_code"], cfg["T_mean"], cfg["T_basis"])
        elif "T_i8" in cfg:
            T = CF.decode_hd(cfg["T_i8"], cfg["T_i8_scale"])
        else:
            T = cfg["T_f16"].astype(np.float32)
        E = CF.decode_e_sparse(cfg["E_values"], fired, ref["E"].shape)
        if fd_pref == "zero":
            fd = np.zeros_like(ref["fd"])
        elif fd_pref.startswith("log"):
            fd = CF.decode_fd_log(cfg["fd_code"], cfg["fd_top"],
                                 int(fd_pref[3:]))
        elif fd_pref == "int8":
            fd = CF.decode_fd_int8(cfg["fd_code"], cfg["fd_top"])
        else:
            fd = CF.decode_fd(cfg["fd_code"])
        hd = CF.decode_hd(cfg["hd_code"], cfg["hd_scale"])
        return dict(T=T, E=E, fd=fd, hd=hd, B=ref["B"])

    rows = {}
    for split in ("val", "test"):
        cfg, ref = build(split)
        path = scratch_path(ds, t_pref, fd_pref)
        path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        np.savez_compressed(path, **cfg)
        w_s = time.perf_counter() - t0
        disk = os.path.getsize(path)
        xz_bytes = sum(lz(v) for v in cfg.values())

        with np.load(path, allow_pickle=False) as z:
            loaded = {k: np.asarray(z[k]) for k in z.files}
        t0 = time.perf_counter()
        rec = decode(loaded, ref)
        dec_s = time.perf_counter() - t0

        if split == "val":
            mae = evaluate(pipe, P_val, yv, vv, s_std, rec)
            base = C0_val
        else:
            Pt = np.asarray(bb["backbone_pred_test"], dtype=np.float32)
            yt, vt = gt(idx_test, Pt)
            base = evaluate(pipe, Pt, yt, vt, s_std, ref)
            mae = evaluate(pipe, Pt, yt, vt, s_std, rec)

        raw_bytes = (ref["T"].nbytes + ref["E"].nbytes + ref["fd"].nbytes
                     + ref["hd"].nbytes + ref["B"].nbytes)
        rows[split] = dict(
            C0_MAE=base, law_MAE=mae, delta=mae - base,
            disk_MB=disk / 1e6, xz_MB=xz_bytes / 1e6,
            raw_MB=raw_bytes / 1e6,
            vs_xz=1.0 - xz_bytes / sum(
                lz(v) for v in (ref["T"], ref["E"], ref["fd"], ref["hd"],
                                ref["B"].astype(np.uint8))),
            write_s=w_s, decode_ms=1e3 * dec_s,
        )
        print(f"    [{split}] C0 {base:.5f}  law {mae:.5f}  "
              f"delta {mae-base:+.5f} | disk {disk/1e6:8.2f} MB  "
              f"xz {xz_bytes/1e6:8.2f} MB (vs xz {100*rows[split]['vs_xz']:+.1f}%)",
              flush=True)
        del cfg, ref, rec, loaded

    cache.close()
    return dict(num_nodes=n_nodes, T_basis_fit_s=fit_s, gate=gate,
                t_pref=t_pref, fd_pref=fd_pref, hd_pref=hd_pref,
                T_isolation=dict(r8=d_r8, f16=d_f16), T_mode=t_mode,
                B_gain=b_gain, B_mode=b_mode, splits=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(EXP))
    ap.add_argument("--gate", type=float, default=0.01)
    ap.add_argument("--t-mode", choices=("auto", "f16", "r8", "i8"),
                    default="auto")
    ap.add_argument("--fd-mode",
                    choices=("f16", "log8", "log6", "log4", "int8", "zero"),
                    default="f16")
    ap.add_argument("--hd-mode", type=int, choices=(16, 8, 6, 4),
                    default=16)
    args = ap.parse_args()

    res = {}
    tag = args.t_mode if args.fd_mode == "f16" else \
        f"{args.t_mode}_{args.fd_mode}"
    out = OUT if tag == "auto" else HERE / f"law_config_{tag}.json"
    for ds in args.datasets:
        try:
            res[ds] = run_dataset(ds, args.gate, args.t_mode, args.fd_mode,
                                  args.hd_mode)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            res[ds] = {"error": f"{type(exc).__name__}: {exc}"}
        old = json.loads(out.read_text("utf-8")) if out.exists() else {}
        old.update(res)
        out.write_text(json.dumps(old, indent=2, ensure_ascii=False),
                       encoding="utf-8")

    print("\n=== law-selected configuration ===")
    print("%-8s %5s %5s %4s %6s %8s %8s %8s %8s"
          % ("ds", "T", "fd", "hd", "B", "val Δ", "test Δ", "vs xz", "MB"))
    for ds, r in res.items():
        if "error" in r:
            print(f"{ds:8s} ERROR {r['error'][:60]}")
            continue
        v, t = r["splits"]["val"], r["splits"]["test"]
        print("%-8s %5s %5s %4d %6s %+8.5f %+8.5f %+7.1f%% %8.2f"
              % (ds, r["T_mode"], r["fd_pref"], r["hd_pref"], r["B_mode"],
                 v["delta"], t["delta"],
                 100 * t["vs_xz"], t["xz_MB"]))
    print(f"\nJSON -> {out}")


if __name__ == "__main__":
    main()
