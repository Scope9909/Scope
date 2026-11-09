"""Semantic codecs vs a strong generic codec: are they additive or redundant?

The earlier measurements compared the field codecs against the *uncompressed*
float32 array, which flatters them: most of the apparent win on E and B is
"the array is 90% zeros", something any generic codec also captures.

This script asks the deployment question instead:
    for each field, does a field-specific codec reduce the size that the best
    generic lossless codec (xz preset 6) already achieves?

    lz(raw field)        = generic-only
    lz(semantic code)    = semantic codec, then the same generic codec
    gain = 1 - lz(code) / lz(raw)

A field only justifies a semantic codec if gain > 0.  Fields where the codec
loses are reported as negative and must be written as such.

Field shapes: T/E/fd/B are (S, H, N, 1); hd is (S, N).  hd is NOT squeezed.

Usage:  python run_semantic_stack.py [PEMS04 PEMS03 ...]
"""

import json
import lzma
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import codec_fields as CF  # noqa: E402

OUT = HERE / "semantic_stack_vs_xz.json"
SPLITS = ("val", "test")


def lz(a, preset=6):
    return len(lzma.compress(np.ascontiguousarray(a).tobytes(),
                             format=lzma.FORMAT_XZ, preset=preset)) / 1e6


def main():
    datasets = sys.argv[1:] or ["PEMS04"]
    res = {}
    for ds in datasets:
        ds_dir = Path(__file__).resolve().parents[1] / "datasets" / ds
        cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                        allow_pickle=False)
        mean, basis = CF.fit_t_basis(cache["retrieval_train"][..., 0])
        per = {}
        for split in SPLITS:
            T = cache[f"retrieval_{split}"][..., 0]
            E = cache[f"exact_retrieval_{split}"][..., 0]
            fd = cache[f"future_dispersion_{split}"][..., 0]
            hd = cache[f"history_distance_{split}"]          # (S, N) - no squeeze
            B = cache[f"exact_available_{split}"][..., 0]
            fired = B.astype(bool)

            codes = {
                "T": [CF.encode_t(T, mean, basis), mean, basis],
                "E": [CF.encode_e_sparse(E, fired), CF.encode_bitmap(fired)],
                "fd": [CF.encode_fd(fd)],
                "hd": list(CF.encode_hd(hd, bits=16)),
                "B": [CF.encode_bitmap(fired)],
            }
            raws = {"T": T, "E": E, "fd": fd, "hd": hd, "B": B}

            print(f"\n=== {ds} / {split}   fired {100 * fired.mean():.1f}% ===")
            print("%-4s %8s %9s %9s %9s %9s %8s"
                  % ("fld", "raw_MB", "lz_raw", "code_MB", "lz_code", "gain", "verdict"))
            rows, tot = {}, {"raw": 0.0, "lz_raw": 0.0, "lz_code": 0.0}
            for k in ("T", "E", "fd", "hd", "B"):
                raw, code = raws[k], codes[k]
                r = lz(raw)
                c_bytes = sum(np.asarray(a).nbytes for a in code) / 1e6
                c = sum(lz(a) for a in code)
                gain = 1 - c / r if r > 0 else 0.0
                verdict = "keep" if gain > 0.01 else ("tie" if gain > -0.01 else "DROP")
                print("%-4s %8.2f %9.2f %9.2f %9.2f %+7.1f%% %8s"
                      % (k, raw.nbytes / 1e6, r, c_bytes, c, 100 * gain, verdict),
                      flush=True)
                rows[k] = dict(raw_MB=round(raw.nbytes / 1e6, 2),
                               lz_raw_MB=round(r, 3),
                               code_MB=round(c_bytes, 2),
                               lz_code_MB=round(c, 3),
                               gain=round(gain, 4), verdict=verdict)
                tot["raw"] += raw.nbytes / 1e6
                tot["lz_raw"] += r
                tot["lz_code"] += c
                del raw, code

            tot["gain_vs_xz"] = 1 - tot["lz_code"] / tot["lz_raw"]
            print("%-4s %8.2f %9.2f %9s %9.2f %+7.1f%%"
                  % ("SUM", tot["raw"], tot["lz_raw"], "",
                     tot["lz_code"], 100 * tot["gain_vs_xz"]), flush=True)
            rows["_total"] = {k: round(v, 3) for k, v in tot.items()}
            # and the shipped baseline for reference
            rows["_total"]["zlib_raw_MB"] = round(
                sum(np.asarray(v).nbytes for v in raws.values()) / 1e6, 2)
            per[split] = rows
        res[ds] = per
        cache.close()

    # merge, never clobber: earlier datasets must survive a re-run of one dataset
    old = {}
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text("utf-8"))
        except Exception:
            old = {}
    old.update(res)
    OUT.write_text(json.dumps(old, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
