"""Which of the five field codecs actually earn their complexity?

Pure arithmetic over semantic_stack_vs_xz.json - no recompression needed.
Every number is lzma(preset 6) bytes, so all configurations are comparable to
"xz only" (the strongest generic lossless baseline).

Configurations
    xz_only   all five fields stored raw, compressed with xz
    A_full    all five fields get their field codec, then xz
    B_Tfd     only T and fd get a codec (the two fields the measurements
              highlight); E/B/hd stay raw
    C_Tfd_hd  B_Tfd plus hd (bit-exact per-node int16, so free)
    D_law     A_full minus the two deployments the law rules out:
              PEMS08 drops T (rank-8 costs +0.0188 > 0.01) and PEMS07 drops
              B (packbits is 15-20% worse than xz there)

Usage: python run_codec_ablation.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "semantic_stack_vs_xz.json"
OUT = HERE / "codec_ablation.json"

FIELDS = ("T", "E", "fd", "hd", "B")
DROP = {"PEMS08": {"T"}, "PEMS07": {"B"}}   # law-derived deployment rules


def main():
    data = json.loads(SRC.read_text("utf-8"))
    out = {}
    print("%-8s %-5s %8s %8s %8s %8s %8s | %8s %8s %8s %8s"
          % ("ds", "split", "xz_only", "A_full", "B_Tfd", "C_Tfdhd", "D_law",
             "A", "B", "C", "D"))
    for ds in sorted(data):
        drop = DROP.get(ds, set())
        tot = {k: 0.0 for k in ("xz_only", "A_full", "B_Tfd", "C_Tfd_hd",
                                "D_law")}
        per = {}
        for split in ("val", "test"):
            f = data[ds][split]
            xz_only = sum(f[k]["lz_raw_MB"] for k in FIELDS)

            def cfg(keep):
                s = 0.0
                for k in FIELDS:
                    s += f[k]["lz_code_MB"] if k in keep else f[k]["lz_raw_MB"]
                return s

            a = cfg(set(FIELDS))
            b = cfg({"T", "fd"})
            c = cfg({"T", "fd", "hd"})
            d = cfg(set(FIELDS) - drop)
            per[split] = dict(xz_only=round(xz_only, 3), A_full=round(a, 3),
                              B_Tfd=round(b, 3), C_Tfd_hd=round(c, 3),
                              D_law=round(d, 3))
            for k, v in (("xz_only", xz_only), ("A_full", a), ("B_Tfd", b),
                         ("C_Tfd_hd", c), ("D_law", d)):
                tot[k] += v
            print("%-8s %-5s %8.2f %8.2f %8.2f %8.2f %8.2f | %+7.1f%% %+7.1f%% "
                  "%+7.1f%% %+7.1f%%"
                  % (ds, split, xz_only, a, b, c, d,
                     100 * (1 - a / xz_only), 100 * (1 - b / xz_only),
                     100 * (1 - c / xz_only), 100 * (1 - d / xz_only)),
                  flush=True)
        per["_total"] = {k: round(v, 3) for k, v in tot.items()}
        for k in ("A_full", "B_Tfd", "C_Tfd_hd", "D_law"):
            per["_total"][f"gain_{k}"] = round(1 - tot[k] / tot["xz_only"], 4)
        per["_total"]["drop"] = sorted(drop)
        out[ds] = per
        print("%-8s %-5s %8.2f %8.2f %8.2f %8.2f %8.2f | %+7.1f%% %+7.1f%% "
              "%+7.1f%% %+7.1f%%"
              % (ds, "SUM", tot["xz_only"], tot["A_full"], tot["B_Tfd"],
                 tot["C_Tfd_hd"], tot["D_law"],
                 100 * (1 - tot["A_full"] / tot["xz_only"]),
                 100 * (1 - tot["B_Tfd"] / tot["xz_only"]),
                 100 * (1 - tot["C_Tfd_hd"] / tot["xz_only"]),
                 100 * (1 - tot["D_law"] / tot["xz_only"])))
        print()
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"JSON -> {OUT}")


if __name__ == "__main__":
    main()
