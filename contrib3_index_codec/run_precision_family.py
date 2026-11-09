"""Same two bytes, different bit layout: fp16 vs per-node int16 under xz.

The Pareto run showed fp16 beats rank-8 on both axes, which leaves the story
resting on "just cast to fp16" - technically true but scientifically thin, and
a reviewer will say so.  This script asks whether a *better* 16-bit layout
exists for the same byte budget:

    fp16              fixed exponent, 10-bit mantissa -- wastes range on
                      traffic that never exceeds ~4 sigma
    per-node int16    one scale per node, 15-bit linear -- spends the whole
                      16 bits on the field's actual dynamic range

Both are 2 bytes/value, so the comparison is purely "which one does xz like
better, and which one is more accurate".  Fields tested: T and fd (the two that
carry the gain).  hd is excluded: it already needs int16 because fp16 overflows.

Reports xz bytes and reconstruction error for each layout.  No pipeline forward
here; accuracy in MAE is measured separately by `run_law_config.py`.

Usage:  python run_precision_family.py [PEMS04 PEMS08 ...]
"""

import json
import lzma
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codec_fields as CF  # noqa: E402

OUT = HERE / "precision_family.json"
SPLITS = ("val", "test")


def lz(a):
    return len(lzma.compress(np.ascontiguousarray(a).tobytes(),
                             format=lzma.FORMAT_XZ, preset=6))


def pernode_int16(x, axis=0):
    """Quantise along ``axis`` with one symmetric scale per column.

    ``x``: (S, H, N) or (S, N) -- the node axis is last.  Returns (int16, scale)
    with scale broadcastable on the node axis.
    """
    x = np.asarray(x, dtype=np.float32)
    absmax = np.abs(x).max(axis=axis)                  # (H,N) or (N,)
    scale = np.where(absmax > 0, absmax / 32767.0, 1.0).astype(np.float32)
    q = np.clip(np.round(x / np.expand_dims(scale, axis)),
                -32767, 32767).astype(np.int16)
    return q, scale


def main():
    datasets = sys.argv[1:] or ["PEMS04", "PEMS08", "PEMS03", "PEMS07"]
    res = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    for ds in datasets:
        ds_dir = Path(__file__).resolve().parents[1] / "datasets" / ds
        cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                        allow_pickle=False)
        per = {}
        for split in SPLITS:
            rows = {}
            for name, arr in (("T", np.ascontiguousarray(
                                   cache[f"retrieval_{split}"][..., 0])),
                              ("fd", np.ascontiguousarray(
                                   cache[f"future_dispersion_{split}"][..., 0]))):
                f16 = arr.astype(np.float16)
                q16, sc16 = pernode_int16(arr)
                q8, sc8 = CF.encode_hd(arr, bits=8)

                lz_f32 = lz(arr)
                lz_f16 = lz(f16)
                lz_i16 = lz(q16) + lz(sc16)
                lz_i8 = lz(q8) + lz(sc8)

                e_f16 = float(np.abs(f16.astype(np.float32) - arr).max())
                e_i16 = float(np.abs(q16.astype(np.float32)
                                     * np.expand_dims(sc16, 0) - arr).max())
                e_i8 = float(np.abs(q8.astype(np.float32)
                                    * np.expand_dims(sc8, 0) - arr).max())
                rows[name] = dict(
                    lz_f32=lz_f32 / 1e6, lz_f16=lz_f16 / 1e6,
                    lz_i16=lz_i16 / 1e6, lz_i8=lz_i8 / 1e6,
                    err_f16=e_f16, err_i16=e_i16, err_i8=e_i8,
                    i16_vs_f16=1.0 - lz_i16 / lz_f16,
                )
                print(f"{ds} {split} {name:2s}: f32 {lz_f32/1e6:7.2f} | "
                      f"f16 {lz_f16/1e6:7.2f} | i16 {lz_i16/1e6:7.2f} "
                      f"({100*rows[name]['i16_vs_f16']:+.1f}% vs f16) | "
                      f"i8 {lz_i8/1e6:6.2f} || maxerr f16 {e_f16:.2e} "
                      f"i16 {e_i16:.2e} i8 {e_i8:.2e}", flush=True)
                del arr, f16, q16, q8
            per[split] = rows
        res[ds] = per
        cache.close()
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
