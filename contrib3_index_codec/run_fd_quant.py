"""fd is non-negative and spans three decades: can a log-domain code beat fp16?

Why this exists
---------------
The final configuration (`run_law_config.py --t-mode i8`) spends 2 bytes per
value on fd.  A linear int8 was already tried and failed catastrophically
(max error 0.93 in z space), and the reason is mechanical rather than bad luck:
fd ranges over 0.33..451 while its median is ~18, so a uniform 8-bit grid has
a step of 1.77 and destroys every small value.  The downstream consumer is
``squash(fd/scale) = log1p(x)/(1+log1p(x))``, which is *most* sensitive
precisely where linear quantisation is worst.

fd is a dispersion (std-like) quantity, so relative error is the natural metric,
and a logarithmic grid makes relative error uniform.  With the per-(h,node)
maximum m as the top of the range:

    t = log1p(x/m) / log 2           in [0, 1], exactly 0 at x = 0
    c = round(levels * t)            uint8 / uint6 / uint4
    x = m * expm1(c/levels * log2)   exact zeros survive

At 8 bits the relative error is ~log2/255 = 0.27% everywhere, versus fp16's
fixed 0.05% *of magnitude* -- which is finer, but fp16 needs twice the bytes.

Layouts compared (all measured after xz, since that is the only honest unit):
    f32             reference
    f16             current configuration
    int8  linear    known failure, kept as the control
    int16 linear    already shown to lose to f16 by 40-57%
    log8 / log6 / log4   the proposals

Reports xz bytes plus error in *z space* (fd / scaler_std), the quantity the
pipeline actually sees, and the error after squash().  No pipeline forward --
MAE is measured by `run_law_config.py --fd-mode logN` once a candidate wins.

Usage:  python run_fd_quant.py [PEMS04 PEMS07 ...]
"""

import json
import lzma
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "fd_quant.json"
SPLITS = ("val", "test")
LOG2 = float(np.log(2.0))


def lz(a):
    return len(lzma.compress(np.ascontiguousarray(a).tobytes(),
                             format=lzma.FORMAT_XZ, preset=6))


def log_code(x, bits):
    """Per-(h,node) log-domain code.  Returns (code, m)."""
    levels = (1 << bits) - 1
    dtype = np.uint8 if bits <= 8 else np.uint16
    x = np.asarray(x, dtype=np.float32)
    m = np.maximum(x.max(axis=0), 1e-9)                       # (H, N)
    t = np.log1p(x / m[None]) / LOG2
    c = np.clip(np.round(t * levels), 0, levels).astype(dtype)
    return c, m.astype(np.float32)


def log_decode(c, m, bits):
    levels = (1 << bits) - 1
    t = c.astype(np.float32) / float(levels) * LOG2
    return (m[None] * np.expm1(t)).astype(np.float32)


def linear_code(x, bits):
    """Per-(h,node) symmetric linear code, matching encode_hd for fd layouts."""
    levels = (1 << (bits - 1)) - 1
    dtype = np.int8 if bits == 8 else np.int16
    x = np.asarray(x, dtype=np.float32)
    absmax = np.abs(x).max(axis=0)
    scale = np.where(absmax > 0, absmax / float(levels), 1.0).astype(np.float32)
    q = np.clip(np.round(x / scale[None]), -levels, levels).astype(dtype)
    return q, scale


def squash(x):
    y = np.log1p(np.clip(x, 0.0, None))
    return y / (1.0 + y)


def main():
    datasets = sys.argv[1:] or ["PEMS04", "PEMS07"]
    res = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    for ds in datasets:
        ds_dir = ROOT / "datasets" / ds
        cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                        allow_pickle=False)
        sc = np.load(ds_dir / f"{ds}_12to12_index_scaler.npz")
        s_std = float(np.asarray(sc["scaler_std"]).ravel()[0]) \
            if "scaler_std" in sc.files else 1.0
        per = {}
        for split in SPLITS:
            fd = np.ascontiguousarray(
                cache[f"future_dispersion_{split}"][..., 0])
            q_ref = squash(fd / s_std)
            z_ref = fd / s_std

            cands = {}
            cands["f32"] = (fd, fd.nbytes, 0.0)
            f16 = fd.astype(np.float16)
            cands["f16"] = (f16.astype(np.float32), lz(f16), 0.0)

            q8, s8 = linear_code(fd, 8)
            cands["int8_lin"] = (q8.astype(np.float32) * s8[None],
                                 lz(q8) + lz(s8), 0.0)
            q16, s16 = linear_code(fd, 16)
            cands["int16_lin"] = (q16.astype(np.float32) * s16[None],
                                  lz(q16) + lz(s16), 0.0)

            for bits in (4, 6, 8):
                c, m = log_code(fd, bits)
                dec = log_decode(c, m, bits)
                cands[f"log{bits}"] = (dec, lz(c) + lz(m), 0.0)

            # global (split-wide) scale variant: drops the per-column m array
            c, m = log_code(fd, 8)
            m_g = np.full_like(m, float(np.max(m)))
            dec_g = log_decode(c, m_g, 8)
            cands["log8_globalscale"] = (dec_g, lz(c) + lz(m_g), 0.0)

            rows = {}
            ref_mb = cands["f16"][1] / 1e6
            print(f"--- {ds} {split}  (scaler_std {s_std:.3f}, "
                  f"z-range 0..{float(z_ref.max()):.2f}) ---")
            print(f"{'layout':18s} {'xz MB':>8s} {'vs f16':>8s} "
                  f"{'z_mae':>9s} {'z_p99':>9s} {'z_max':>9s} "
                  f"{'sq_err':>9s} {'rel_p99':>8s}")
            for name, (dec, nbytes, _) in cands.items():
                z = dec / s_std
                dz = np.abs(z - z_ref)
                sq = np.abs(squash(z) - q_ref)
                nz = np.abs(fd) > 0
                rel = np.abs(dec[nz] - fd[nz]) / np.abs(fd[nz])
                row = dict(
                    xz_MB=nbytes / 1e6 if name != "f32" else nbytes / 1e6,
                    z_mae=float(dz.mean()), z_p99=float(np.percentile(dz, 99)),
                    z_max=float(dz.max()),
                    squash_mae=float(sq.mean()), squash_max=float(sq.max()),
                    rel_p99=float(np.percentile(rel, 99)),
                )
                rows[name] = row
                print(f"{name:18s} {row['xz_MB']:8.2f} "
                      f"{100*(row['xz_MB']/ref_mb-1):+7.1f}% "
                      f"{row['z_mae']:9.5f} {row['z_p99']:9.5f} "
                      f"{row['z_max']:9.5f} {row['squash_mae']:9.6f} "
                      f"{row['rel_p99']:8.4f}", flush=True)
            per[split] = rows
            del fd, q_ref, z_ref, cands
        res[ds] = per
        cache.close()
        sc.close()
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
