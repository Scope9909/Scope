"""Does the candidate-index codec beat xz too, or only beat int32?

The chain accounting quotes -81.25% for the index segment, but that is measured
against raw int32.  A reviewer asks the same question as for the cache: what
does xz alone do to int32 endpoints?

    int32 raw | xz(int32) | 6-bit TOD offsets | xz(6-bit)

Usage: python run_index_vs_xz.py [PEMS04 ...]
"""

import json
import lzma
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_index_codec_audit as R  # noqa: E402

OUT = HERE / "index_vs_xz.json"


def lz(a, preset=6):
    return len(lzma.compress(np.ascontiguousarray(a).tobytes(),
                             format=lzma.FORMAT_XZ, preset=preset)) / 1e6


def main():
    datasets = sys.argv[1:] or ["PEMS04"]
    res = {}
    for ds in datasets:
        paths = R.dataset_paths(ds)
        with np.load(paths["index_scaler"], allow_pickle=False) as z:
            idx_train = np.asarray(z["idx_train"], dtype=np.int64)
            idx_val = np.asarray(z["idx_val"], dtype=np.int64)
            idx_test = np.asarray(z["idx_test"], dtype=np.int64)
        with np.load(paths["time"], allow_pickle=False) as z:
            time_data = np.asarray(z["time"])
        num_steps = int(time_data.shape[0])
        train_tod, groups = R.build_tod_groups(time_data, idx_train)
        gmax = max(g.size for g in groups.values())
        sentinel = gmax
        bits = int(np.ceil(np.log2(gmax + 1)))
        table = R.build_position_table(num_steps, groups)

        with np.load(paths["indices"], allow_pickle=False) as z:
            keys = [k for k in z.files
                    if k.startswith("candidate_index_")
                    and (k.endswith("val") or k.endswith("test"))]
            print(f"\n=== {ds}  groups<= {gmax} ({bits} bit) ===")
            print("%-20s %10s %10s %10s %10s %9s %9s"
                  % ("split", "raw_i32", "xz_i32", "pack6", "xz_pack6",
                     "6bit/xz", "pack6/xz6"))
            per, tot = {}, dict(raw=0.0, xz=0.0, pack=0.0, xz_pack=0.0)
            for k in keys:
                a = np.asarray(z[k])
                qidx = idx_val if k.endswith("val") else idx_test
                qtod = np.asarray(time_data[qidx, 0], dtype=np.int64)
                raw = a.nbytes / 1e6
                xz_i32 = lz(a)
                arr = a[..., 0] if a.ndim == 4 else a
                code = R.encode_t2(arr, table, sentinel)        # uint8 offsets
                packed = R.bitpack(np.ascontiguousarray(
                    code.transpose(0, 2, 1)), bits)             # (Q,N,cols)
                pbytes = packed.nbytes / 1e6
                xz_packed = lz(packed)
                per[k] = dict(raw=round(raw, 2), xz_i32=round(xz_i32, 3),
                              pack6=round(pbytes, 2), xz_pack6=round(xz_packed, 3),
                              bits=bits)
                print("%-20s %10.2f %10.2f %10.2f %10.2f %+8.1f%% %+8.1f%%"
                      % (k, raw, xz_i32, pbytes, xz_packed,
                         100 * (1 - pbytes / xz_i32),
                         100 * (1 - xz_packed / xz_i32)), flush=True)
                for key, v in (("raw", raw), ("xz", xz_i32),
                               ("pack", pbytes), ("xz_pack", xz_packed)):
                    tot[key] += v
                del a, code, packed
        tot["pack_vs_xz"] = 1 - tot["pack"] / tot["xz"]
        tot["xzpack_vs_xz"] = 1 - tot["xz_pack"] / tot["xz"]
        print("%-20s %10.2f %10.2f %10.2f %10.2f %+8.1f%% %+8.1f%%"
              % ("TOTAL", tot["raw"], tot["xz"], tot["pack"], tot["xz_pack"],
                 100 * tot["pack_vs_xz"], 100 * tot["xzpack_vs_xz"]))
        per["_total"] = {k: round(float(v), 3) for k, v in tot.items()}
        res[ds] = per

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
