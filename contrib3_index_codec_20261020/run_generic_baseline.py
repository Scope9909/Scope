"""Is FCL beating *generic lossless* compression, or only beating zlib?

The array-byte -> file-byte correction showed that ``savez_compressed`` (zlib,
level 6) already removes 2.6-3.8x from the raw cache.  A reviewer will ask the
obvious follow-up: if a better generic codec (LZMA) also removes that
redundancy, then FCL's on-disk win may be an artefact of comparing against a
weak baseline rather than a property of the semantic codecs.

This script answers that for the two payload splits that a deployment actually
serves (val + test), on the same arrays:

    zlib  (np.savez_compressed, the shipped format)
    lzma  (xz, preset 6 - a much stronger generic lossless codec)
    fcl   (the field codecs, measured by run_physical_artifacts.py)

If ``lzma(all_f32) <= fcl``, the storage claim collapses to "we replaced one
generic codec with another" and must NOT be written as a compression result.
Scope is val+test only (train payload is training/replay infrastructure).
"""

import io
import json
import lzma
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "generic_baseline_val_test.json"
SPLITS = ("val", "test")


def zlib_size(arrays):
    """Exactly the shipped format: np.savez_compressed (zlib level 6)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getbuffer().nbytes


def lzma_size(arrays, preset=6):
    """xz container, preset 6 - a much stronger generic lossless codec."""
    total = 0
    for k, v in arrays.items():
        a = np.ascontiguousarray(v)
        total += len(lzma.compress(a.tobytes(), format=lzma.FORMAT_XZ,
                                   preset=preset))
    return total


def main(datasets):
    phys = json.loads((HERE / "physical_artifacts.json").read_text("utf-8"))
    out = {}
    for ds in datasets:
        ds_dir = Path(__file__).resolve().parents[1] / "datasets" / ds
        cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                        allow_pickle=False)
        per = {}
        tot = {"zlib_MB": 0.0, "lzma_MB": 0.0, "fcl_file_MB": 0.0,
               "array_MB": 0.0}
        for split in SPLITS:
            arrays = {
                "T": cache[f"retrieval_{split}"],
                "E": cache[f"exact_retrieval_{split}"],
                "fd": cache[f"future_dispersion_{split}"],
                "hd": cache[f"history_distance_{split}"],
                "B": cache[f"exact_available_{split}"],
            }
            arr_mb = sum(v.nbytes for v in arrays.values()) / 1e6
            z_mb = zlib_size(arrays) / 1e6
            l_mb = lzma_size(arrays) / 1e6
            fcl_mb = phys[ds]["per_split_file_MB"][split]["fcl"]
            per[split] = {"array_MB": round(arr_mb, 2),
                          "zlib_MB": round(z_mb, 2),
                          "lzma_MB": round(l_mb, 2),
                          "fcl_file_MB": round(fcl_mb, 2)}
            print("%-7s %-5s array %8.1f  zlib %8.1f  lzma %8.1f  fcl %8.1f"
                  "   | fcl vs zlib %+6.1f%%  fcl vs lzma %+6.1f%%"
                  % (ds, split, arr_mb, z_mb, l_mb, fcl_mb,
                     100 * (1 - fcl_mb / z_mb), 100 * (1 - fcl_mb / l_mb)),
                  flush=True)
            for k, v in (("zlib_MB", z_mb), ("lzma_MB", l_mb),
                         ("fcl_file_MB", fcl_mb), ("array_MB", arr_mb)):
                tot[k] += v
            del arrays
        per["total"] = {k: round(v, 2) for k, v in tot.items()}
        per["fcl_vs_zlib"] = 1.0 - tot["fcl_file_MB"] / tot["zlib_MB"]
        per["fcl_vs_lzma"] = 1.0 - tot["fcl_file_MB"] / tot["lzma_MB"]
        per["lzma_vs_zlib"] = 1.0 - tot["lzma_MB"] / tot["zlib_MB"]
        out[ds] = per
        print("%-7s TOTAL array %8.1f  zlib %8.1f  lzma %8.1f  fcl %8.1f"
              "   | fcl vs zlib %+6.1f%%  fcl vs lzma %+6.1f%%"
              % (ds, tot["array_MB"], tot["zlib_MB"], tot["lzma_MB"],
                 tot["fcl_file_MB"], 100 * per["fcl_vs_zlib"],
                 100 * per["fcl_vs_lzma"]), flush=True)
        cache.close()
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["PEMS04"])
