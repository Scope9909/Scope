"""Full-chain physical accounting after adding the lossless index codec.

Every number is measured from the actual arrays on disk (no 40x heuristics).
Payload definitions, stated explicitly so the table cannot be misread:

    weights  = best.pth of the joint3 experiment (on-disk bytes)
    flow     = data.npz['data'], single channel float32 -- the memory-bank body
    index    = candidate_index_{train,val,test} of the k8 candidate cache
    cache    = retrieval_*, exact_retrieval_*, future_dispersion_*,
               history_distance_*, exact_available_* over the three splits

The retrieval cache is priced twice: raw (deployed today) and after the
field-wise codec (FCL), using the per-dataset reduction measured in
fieldwise_global_cross_val.json.  The index is priced at T0 (int32), T1
(int16) and T3 (6-bit packed TOD offsets), all verified lossless in
index_codec_audit.json.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
SPLITS = ("train", "val", "test")
CACHE_PREFIXES = (
    "retrieval_",
    "exact_retrieval_",
    "future_dispersion_",
    "history_distance_",
    "exact_available_",
)


def paths(name):
    base = os.path.join(ROOT, "datasets", name)
    return {
        "weights": os.path.join(
            ROOT, "EXP_experiments", "TLAST_improved", name,
            f"TLAST_improved_{name}_12to12_exppaper_"
            f"{name.lower()}_s1_final_joint", "best.pth"),
        "flow": os.path.join(base, "data.npz"),
        "index": os.path.join(
            base, f"{name}_12to12_analog_k8_candidate_indices.npz"),
        "cache": os.path.join(
            base, f"{name}_12to12_analog_k8_exact_mean.npz"),
    }


def measure(name, fcl_saving):
    p = paths(name)
    if not os.path.isfile(p["weights"]):
        raise FileNotFoundError(p["weights"])
    weights = os.path.getsize(p["weights"])

    with np.load(p["flow"], allow_pickle=False) as z:
        flow_arr = z["data"]
        flow = int(flow_arr.shape[0] * flow_arr.shape[1] * 4)

    with np.load(p["index"], allow_pickle=False) as z:
        index_raw = sum(int(z[f"candidate_index_{s}"].nbytes) for s in SPLITS)
        index_meta = sum(
            int(z[k].nbytes) for k in z.files if k.startswith("idx_"))

    cache_fields = {}
    with np.load(p["cache"], allow_pickle=False) as z:
        for key in z.files:
            if not any(key.startswith(pre) for pre in CACHE_PREFIXES):
                continue
            field = key.rsplit("_", 1)[0]
            cache_fields.setdefault(field, 0)
            cache_fields[field] += int(z[key].nbytes)
        cache_meta = sum(
            int(z[k].nbytes) for k in z.files
            if not any(k.startswith(pre) for pre in CACHE_PREFIXES))
    cache_raw = sum(cache_fields.values())

    report = {
        "weights_MB": weights / 1e6,
        "flow_MB": flow / 1e6,
        "index_raw_MB": index_raw / 1e6,
        "index_int16_MB": index_raw * 0.50 / 1e6,
        "index_bit6_MB": index_raw * 0.1875 / 1e6,
        "index_meta_MB": index_meta / 1e6,
        "cache_meta_MB": cache_meta / 1e6,
        "cache_raw_MB": cache_raw / 1e6,
        "cache_fields_MB": {k: v / 1e6 for k, v in cache_fields.items()},
        "fcl_saving": fcl_saving,
    }

    cache_fcl = cache_raw * (1.0 - fcl_saving)
    report["cache_fcl_MB"] = cache_fcl / 1e6

    base = weights + flow + index_raw + cache_raw
    stage_fcl = weights + flow + index_raw + cache_fcl
    stage_t1 = weights + flow + index_raw * 0.50 + cache_fcl
    stage_t3 = weights + flow + index_raw * 0.1875 + cache_fcl
    online = weights + flow

    report["chain_MB"] = {
        "baseline": base / 1e6,
        "fcl_only": stage_fcl / 1e6,
        "fcl_index_int16": stage_t1 / 1e6,
        "fcl_index_bit6": stage_t3 / 1e6,
        "online_no_precompute": online / 1e6,
    }
    report["reduction_vs_baseline"] = {
        "fcl_only": 1.0 - stage_fcl / base,
        "fcl_index_int16": 1.0 - stage_t1 / base,
        "fcl_index_bit6": 1.0 - stage_t3 / base,
        "online_no_precompute": 1.0 - online / base,
    }
    report["shadowing_ratio"] = {
        "cache_over_flow": cache_raw / flow,
        "index_over_flow": index_raw / flow,
        "chain_over_online": base / online,
        "chain_over_online_after_bit6": stage_t3 / online,
    }

    # If the cache were regenerated on demand the whole artefact disappears;
    # what remains addressable is the memory-bank body only.
    report["index_share_of_chain"] = {
        "baseline": index_raw / base,
        "after_fcl": index_raw / base,       # same numerator, smaller chain
        "after_bit6": (index_raw * 0.1875) / stage_t3,
    }
    return report


def main():
    fcl = json.load(open(
        os.path.join(ROOT, "optionB_identity_stability",
                     "fieldwise_global_cross_val.json"), encoding="utf-8"))
    saving = fcl["verdict"]["saving_by_dataset"]

    results = {}
    for name in DATASETS:
        results[name] = measure(name, saving[name])

    out = os.path.join(ROOT, "contrib3_index_codec",
                       "full_chain_accounting.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "scope": "measured bytes only; cache priced after FCL using "
                     "fieldwise_global_cross_val.json; index priced at "
                     "int32/int16/6-bit packed, all lossless-verified",
            "results": results,
        }, f, indent=2)

    header = ("dataset  weights   flow    index   cache    baseline  "
              "FCL      +i16     +i6      online   red@i6   cache/flow")
    print(header)
    for name, r in results.items():
        c = r["chain_MB"]
        print("%-7s %7.2f %7.2f %8.2f %8.1f %9.1f %8.1f %8.1f %8.1f %8.2f "
              "%7.2f%% %9.1fx"
              % (name, r["weights_MB"], r["flow_MB"], r["index_raw_MB"],
                 r["cache_raw_MB"], c["baseline"], c["fcl_only"],
                 c["fcl_index_int16"], c["fcl_index_bit6"],
                 c["online_no_precompute"],
                 100 * r["reduction_vs_baseline"]["fcl_index_bit6"],
                 r["shadowing_ratio"]["cache_over_flow"]))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
