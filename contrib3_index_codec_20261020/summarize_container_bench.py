"""Phase C accounting -- same container, real file bytes.

``raw`` is a float32 payload written through the *same* self-describing xz
container, so it is the pure-xz baseline measured in the same file format as
ours.  Everything below is therefore an apples-to-apples comparison of real
files on disk (stat().st_size), not a sum of theoretical per-array sizes.

Emits container_bench_summary.json + a markdown block on stdout.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "container_bench.json"
OUT = HERE / "container_bench_summary.json"

doc = json.loads(SRC.read_text("utf-8"))
DS_ORDER = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
METHODS = ["raw", "naive8", "ours_full", "ours_min"]

rows = []
tot = {m: {"cache": 0.0, "index_raw": 0.0, "index_six": 0.0,
           "cache_read": 0.0, "cache_dec": 0.0,
           "raw_read": 0.0, "raw_dec": 0.0,
           "six_read": 0.0, "six_dec": 0.0,
           "peak_cache": 0.0, "peak_raw": 0.0, "peak_six": 0.0,
           "d_val": [], "d_test": []}
       for m in METHODS}

for ds in DS_ORDER:
    d = doc[ds]
    for m in METHODS:
        c_val = d["cache"][f"{m}/val"]
        c_test = d["cache"][f"{m}/test"]
        i_raw_val = d["index"]["raw/val"]
        i_raw_test = d["index"]["raw/test"]
        i_six_val = d["index"]["sixbit/val"]
        i_six_test = d["index"]["sixbit/test"]
        cache_mb = c_val["size_MB"] + c_test["size_MB"]
        idx_raw_mb = i_raw_val["size_MB"] + i_raw_test["size_MB"]
        idx_six_mb = i_six_val["size_MB"] + i_six_test["size_MB"]
        t = tot[m]
        t["cache"] += cache_mb
        t["index_raw"] += idx_raw_mb
        t["index_six"] += idx_six_mb
        t["cache_read"] += c_val["read_ms"] + c_test["read_ms"]
        t["cache_dec"] += c_val["decode_ms"] + c_test["decode_ms"]
        t["raw_read"] += i_raw_val["read_ms"] + i_raw_test["read_ms"]
        t["raw_dec"] += i_raw_val["decode_ms"] + i_raw_test["decode_ms"]
        t["six_read"] += i_six_val["read_ms"] + i_six_test["read_ms"]
        t["six_dec"] += i_six_val["decode_ms"] + i_six_test["decode_ms"]
        t["peak_cache"] = max(t["peak_cache"], c_val["peak_delta_MB"],
                              c_test["peak_delta_MB"])
        t["peak_raw"] = max(t["peak_raw"], i_raw_val["peak_delta_MB"],
                            i_raw_test["peak_delta_MB"])
        t["peak_six"] = max(t["peak_six"], i_six_val["peak_delta_MB"],
                            i_six_test["peak_delta_MB"])
        rows.append(dict(
            ds=ds, method=m,
            cache_MB=cache_mb,
            idx_raw_MB=idx_raw_mb, idx_six_MB=idx_six_mb,
            val_MAE=c_val["MAE"], test_MAE=c_test["MAE"],
            read_ms=c_val["read_ms"] + c_test["read_ms"],
            dec_ms=c_val["decode_ms"] + c_test["decode_ms"],
            peak_MB=max(c_val["peak_delta_MB"], c_test["peak_delta_MB"])))

# deltas are relative to the raw container of the same dataset
delta = {}
for ds in DS_ORDER:
    base_val = doc[ds]["cache"]["raw/val"]["MAE"]
    base_test = doc[ds]["cache"]["raw/test"]["MAE"]
    delta[ds] = {m: dict(val=doc[ds]["cache"][f"{m}/val"]["MAE"] - base_val,
                         test=doc[ds]["cache"][f"{m}/test"]["MAE"] - base_test)
                 for m in METHODS}
    delta[ds]["C0"] = dict(val=base_val, test=base_test)
    for m in METHODS:
        tot[m]["d_val"].append(delta[ds][m]["val"])
        tot[m]["d_test"].append(delta[ds][m]["test"])

BASE = tot["raw"]["cache"] + tot["raw"]["index_raw"]


def pct(saved):
    return 100.0 * (1.0 - saved / BASE)


def chain(m, idx):
    """idx in {'six','raw'}: which candidate-index container this chain ships."""
    idx_mb = tot[m]["index_six" if idx == "six" else "index_raw"]
    idx_read = tot[m]["six_read" if idx == "six" else "raw_read"]
    idx_dec = tot[m]["six_dec" if idx == "six" else "raw_dec"]
    idx_peak = tot[m]["peak_six" if idx == "six" else "peak_raw"]
    total = tot[m]["cache"] + idx_mb
    return dict(
        cache_MB=tot[m]["cache"], index_MB=idx_mb, total_MB=total,
        saving_pct=pct(total), index_variant="6-bit" if idx == "six" else "xz",
        ms=tot[m]["cache_read"] + tot[m]["cache_dec"] + idx_read + idx_dec,
        peak_MB=max(tot[m]["peak_cache"], idx_peak))


chains = {}
for m in ("raw", "naive8"):
    chains[f"{m}+xz-index"] = chain(m, "raw")
chains["naive8+6bit-index"] = chain("naive8", "six")
for m in ("ours_full", "ours_min"):
    chains[f"{m}+xz-index"] = chain(m, "raw")
    chains[f"{m}+6bit-index"] = chain(m, "six")

summary = dict(
    baseline_total_MB=BASE,
    baseline_cache_MB=tot["raw"]["cache"],
    baseline_index_MB=tot["raw"]["index_raw"],
    per_dataset=rows, delta_MAE=delta, chains=chains,
    max_test_delta={m: max(abs(v) for v in tot[m]["d_test"]) for m in METHODS},
    max_val_delta={m: max(abs(v) for v in tot[m]["d_val"]) for m in METHODS},
    total_read_decode_ms={m: tot[m]["cache_read"] + tot[m]["cache_dec"]
                          for m in METHODS},
)
OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")

# ------------------------------------------------------------------ report --
print(f"baseline (raw container = pure xz, same format): "
      f"{BASE:.2f} MB  (cache {tot['raw']['cache']:.2f} + index "
      f"{tot['raw']['index_raw']:.2f})\n")
print("| chain | cache MB | index MB | total MB | saving vs xz | max |dMAE| val "
      "| max |dMAE| test | read+dec ms | peak MB |")
print("|---|---|---|---|---|---|---|---|---|")
for k, v in chains.items():
    m = k.split("+")[0]
    print(f"| {k} | {v['cache_MB']:.2f} | {v['index_MB']:.2f} | {v['total_MB']:.2f} "
          f"| -{v['saving_pct']:.1f}% | {summary['max_val_delta'][m]:.5f} "
          f"| {summary['max_test_delta'][m]:.5f} | {v['ms']:.0f} "
          f"| {v['peak_MB']:.0f} |")
print()
print("per-dataset cache MAE delta vs the raw container")
print("| ds | C0 val | C0 test | naive8 val | naive8 test | ours_full val | ours_full test | ours_min val | ours_min test |")
print("|---|---|---|---|---|---|---|---|---|")
for ds in DS_ORDER:
    b = delta[ds]["C0"]
    f = "| %s | %.5f | %.5f " % (ds, b["val"], b["test"])
    for m in ("naive8", "ours_full", "ours_min"):
        f += "| %+.5f | %+.5f " % (delta[ds][m]["val"], delta[ds][m]["test"])
    print(f + "|")
print()
print("index segment on its own (8 splits = 4 datasets x val/test)")
print("| variant | MB | read ms | decode ms | peak MB (max split) |")
print("|---|---|---|---|---|")
for variant, key in (("xz (int32)", "raw"), ("6-bit TOD offsets", "sixbit")):
    mb = sum(doc[ds]["index"][f"{key}/{s}"]["size_MB"]
             for ds in DS_ORDER for s in ("val", "test"))
    rd = sum(doc[ds]["index"][f"{key}/{s}"]["read_ms"]
             for ds in DS_ORDER for s in ("val", "test"))
    dc = sum(doc[ds]["index"][f"{key}/{s}"]["decode_ms"]
             for ds in DS_ORDER for s in ("val", "test"))
    pk = max(doc[ds]["index"][f"{key}/{s}"]["peak_delta_MB"]
             for ds in DS_ORDER for s in ("val", "test"))
    print(f"| {variant} | {mb:.2f} | {rd:.0f} | {dc:.0f} | {pk:.0f} |")
print()
print("wall time, read+decode over val+test (ms)")
for m in METHODS:
    print(f"  {m:10s} {summary['total_read_decode_ms'][m]:9.0f}")
print(f"\nJSON -> {OUT}")
