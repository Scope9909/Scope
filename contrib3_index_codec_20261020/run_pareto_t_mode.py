"""Pareto front of the T-field codec: rank-8 vs bare fp16, on real bytes.

The law says "use rank-8 only while its cost stays under the gate".  Two of the
four datasets sit close to that gate (PEMS03 +0.0078, PEMS08 +0.0180), so the
interesting question is what the *storage* side gives up when you switch T from
rank-8 to fp16.  `run_law_config.py` already measured the accuracy of both
modes in isolation (T isolation r8 / f16); this script measures only bytes, so
the two together give a real accuracy-vs-storage Pareto table.

For each dataset it reports, with every other field held at the law-selected
configuration:
    T = rank-8 code (+ mean + basis)   -> xz bytes
    T = fp16                           -> xz bytes
    T = float32                        -> xz bytes
and the byte delta between them.

Usage:  python run_pareto_t_mode.py [PEMS03 PEMS04 PEMS07 PEMS08]
"""

import json
import lzma
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codec_fields as CF  # noqa: E402

OUT = HERE / "pareto_t_mode.json"
LAW = HERE / "law_config.json"
SPLITS = ("val", "test")


def lz(a):
    return len(lzma.compress(np.ascontiguousarray(a).tobytes(),
                             format=lzma.FORMAT_XZ, preset=6))


def main():
    datasets = sys.argv[1:] or ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
    res = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    for ds in datasets:
        ds_dir = Path(__file__).resolve().parents[1] / "datasets" / ds
        cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                        allow_pickle=False)
        mean, basis = CF.fit_t_basis(cache["retrieval_train"][..., 0])
        per = {}
        for split in SPLITS:
            T = np.ascontiguousarray(cache[f"retrieval_{split}"][..., 0])
            E = np.ascontiguousarray(cache[f"exact_retrieval_{split}"][..., 0])
            fd = np.ascontiguousarray(cache[f"future_dispersion_{split}"][..., 0])
            hd = np.ascontiguousarray(cache[f"history_distance_{split}"])
            B = cache[f"exact_available_{split}"][..., 0]
            fired = B.astype(bool)

            # the four non-T fields, at their law-selected codecs
            others = {
                "E_values": CF.encode_e_sparse(E, fired),
                "bitmap": CF.encode_bitmap(fired),
                "fd_code": CF.encode_fd(fd),
            }
            others["hd_code"], others["hd_scale"] = CF.encode_hd(hd, bits=16)
            lz_others = sum(lz(v) for v in others.values())
            lz_B_raw = lz(B.astype(np.uint8))

            t_r8 = CF.encode_t(T, mean, basis)
            lz_r8 = lz(t_r8) + lz(mean) + lz(basis)
            lz_f16 = lz(T.astype(np.float16))
            lz_f32 = lz(T)

            per[split] = dict(
                lz_others=lz_others / 1e6, lz_B_raw=lz_B_raw / 1e6,
                lz_T_r8=lz_r8 / 1e6, lz_T_f16=lz_f16 / 1e6,
                lz_T_f32=lz_f32 / 1e6,
                total_r8=(lz_others + lz_B_raw + lz_r8) / 1e6,
                total_f16=(lz_others + lz_B_raw + lz_f16) / 1e6,
                total_f32=(lz_others + lz_B_raw + lz_f32) / 1e6,
            )
            print(f"{ds} {split}: T r8 {lz_r8/1e6:7.2f}  f16 {lz_f16/1e6:7.2f}  "
                  f"f32 {lz_f32/1e6:7.2f} MB | total r8 {per[split]['total_r8']:7.2f} "
                  f"f16 {per[split]['total_f16']:7.2f}  "
                  f"-> f16 keeps {100*(1-per[split]['total_f16']/per[split]['total_r8']):.1f}% "
                  f"of the r8 saving", flush=True)
            del T, E, fd, hd, B, fired, others, t_r8
        res[ds] = per
        cache.close()
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
