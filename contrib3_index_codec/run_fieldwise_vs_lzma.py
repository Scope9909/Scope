"""Where does FCL actually beat a strong generic codec, and at what decode cost?

run_generic_baseline.py answered the aggregate question: FCL beats xz by only
10.8-17.6% on val+test.  That is too thin to carry a storage claim on its own,
so this script decomposes it field by field.

1. PER-FIELD bytes: raw array | zlib | lzma | field codec payload.
2. PER-FIELD decode: lzma must inflate the whole field before any lookup;
   the sparse payload code only touches the fired slots.
3. FIRING RATE: what fraction of the E field a deployment actually reads.

Usage:  python run_fieldwise_vs_lzma.py [PEMS04] [--splits=val,test]
"""

import io
import json
import lzma
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import codec_fields as CF  # noqa: E402

OUT = HERE / "fieldwise_vs_lzma.json"


def zlib_bytes(a):
    buf = io.BytesIO()
    np.savez_compressed(buf, x=a)
    return buf.getbuffer().nbytes


def lzma_blob(a):
    return lzma.compress(np.ascontiguousarray(a).tobytes(),
                         format=lzma.FORMAT_XZ, preset=6)


def timeit(fn, repeat=3):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3, out


def payload_mb(*arrs):
    return sum(np.asarray(a).nbytes for a in arrs) / 1e6


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    datasets = args or ["PEMS04"]
    splits = ("val", "test")
    for a in sys.argv[1:]:
        if a.startswith("--splits="):
            splits = tuple(a.split("=", 1)[1].split(","))

    res = {}
    for ds in datasets:
        ds_dir = Path(__file__).resolve().parents[1] / "datasets" / ds
        cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                        allow_pickle=False)
        mean, basis = CF.fit_t_basis(cache["retrieval_train"][..., 0])
        per = {}
        for split in splits:
            T = cache[f"retrieval_{split}"][..., 0]
            E = cache[f"exact_retrieval_{split}"][..., 0]
            fd = cache[f"future_dispersion_{split}"][..., 0]
            hd = cache[f"history_distance_{split}"]
            B = cache[f"exact_available_{split}"][..., 0]
            fired = B.astype(bool)
            print(f"\n=== {ds} / {split}   fired={100 * fired.mean():.1f}% ===")
            print("%-4s %-18s %8s %8s %8s %8s %9s %10s %10s"
                  % ("fld", "shape", "raw_MB", "zlib", "lzma", "FCL",
                     "vs_lzma", "lzma_dms", "fcl_dms"))
            rows = {}

            def report(tag, shape, raw, z, l, fcl, lms, fms, extra=None):
                print("%-4s %-18s %8.1f %8.1f %8.1f %8.1f %+8.1f%% %10.1f %10.1f"
                      % (tag, str(shape), raw, z, l, fcl,
                         100 * (1 - fcl / l), lms, fms), flush=True)
                d = dict(shape=list(shape), raw_MB=round(raw, 2),
                         zlib_MB=round(z, 2), lzma_MB=round(l, 2),
                         fcl_MB=round(fcl, 2), vs_lzma=1 - fcl / l,
                         lzma_dec_ms=round(lms, 1), fcl_dec_ms=round(fms, 1))
                if extra:
                    d.update(extra)
                rows[tag] = d

            # T: rank-8 PCA over horizon, fp16 codes
            code = CF.encode_t(T, mean, basis)
            raw, z = T.nbytes / 1e6, zlib_bytes(T) / 1e6
            bl = lzma_blob(T)
            l = len(bl) / 1e6
            fcl = payload_mb(code) + payload_mb(mean, basis)
            lms, _ = timeit(lambda: np.frombuffer(
                lzma.decompress(bl), dtype=np.float32).reshape(T.shape))
            fms, dec = timeit(lambda: CF.decode_t(code, mean, basis))
            report("T", T.shape, raw, z, l, fcl, lms, fms,
                   dict(max_abs_err=float(np.max(np.abs(dec - T)))))
            del bl, dec

            # E: shared bitmap + compacted fp16 payload
            vals = CF.encode_e_sparse(E, fired)
            raw, z = E.nbytes / 1e6, zlib_bytes(E) / 1e6
            bl = lzma_blob(E)
            l = len(bl) / 1e6
            fcl = payload_mb(vals)
            lms, _ = timeit(lambda: np.frombuffer(
                lzma.decompress(bl), dtype=np.float32).reshape(E.shape))
            fms, dec = timeit(lambda: CF.decode_e_sparse(vals, fired, E.shape))
            report("E", E.shape, raw, z, l, fcl, lms, fms,
                   dict(max_abs_err=float(np.max(np.abs(dec - E))),
                        payload_slots=int(vals.size), fired=int(fired.sum())))
            del bl, dec, vals

            # fd: fp16
            h = CF.encode_fd(fd)
            raw, z = fd.nbytes / 1e6, zlib_bytes(fd) / 1e6
            bl = lzma_blob(fd)
            l = len(bl) / 1e6
            lms, _ = timeit(lambda: np.frombuffer(
                lzma.decompress(bl), dtype=np.float32).reshape(fd.shape))
            fms, dec = timeit(lambda: CF.decode_fd(h))
            report("fd", fd.shape, raw, z, l, payload_mb(h), lms, fms,
                   dict(max_abs_err=float(np.max(np.abs(dec - fd)))))
            del bl, dec

            # hd: per-node int16 (fp16 overflows: |x| up to 1e5)
            q, sc = CF.encode_hd(hd, bits=16)
            raw, z = hd.nbytes / 1e6, zlib_bytes(hd) / 1e6
            bl = lzma_blob(hd)
            l = len(bl) / 1e6
            lms, _ = timeit(lambda: np.frombuffer(
                lzma.decompress(bl), dtype=np.float32).reshape(hd.shape))
            fms, dec = timeit(lambda: CF.decode_hd(q, sc))
            report("hd", hd.shape, raw, z, l, payload_mb(q, sc), lms, fms,
                   dict(max_abs_err=float(np.max(np.abs(dec - hd))),
                        raw_absmax=float(np.abs(hd).max())))
            del bl, dec

            # B: packed bitmap
            bits = CF.encode_bitmap(fired)
            raw, z = B.nbytes / 1e6, zlib_bytes(B) / 1e6
            bl = lzma_blob(B)
            l = len(bl) / 1e6
            lms, _ = timeit(lambda: np.frombuffer(
                lzma.decompress(bl), dtype=B.dtype).reshape(B.shape))
            fms, dec = timeit(lambda: CF.decode_bitmap(bits, B.size))
            report("B", B.shape, raw, z, l, payload_mb(bits), lms, fms,
                   dict(exact=bool(np.array_equal(dec, fired))))
            del bits, bl, dec

            tot = {k: sum(r[k] for r in rows.values())
                   for k in ("raw_MB", "zlib_MB", "lzma_MB", "fcl_MB",
                             "lzma_dec_ms", "fcl_dec_ms")}
            rows["_total"] = tot
            print("%-4s %-18s %8.1f %8.1f %8.1f %8.1f %+8.1f%% %10.1f %10.1f"
                  % ("SUM", "", tot["raw_MB"], tot["zlib_MB"], tot["lzma_MB"],
                     tot["fcl_MB"], 100 * (1 - tot["fcl_MB"] / tot["lzma_MB"]),
                     tot["lzma_dec_ms"], tot["fcl_dec_ms"]), flush=True)

            per[split] = rows
        res[ds] = per
        cache.close()

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
