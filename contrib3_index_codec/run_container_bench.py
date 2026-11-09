"""Phase C -- the honest cost comparison, in one file format.

The previous headline ("-64.4%") was computed as *sum of per-array xz sizes* on
both sides.  That is internally consistent but it is not a file you can ship:
the actual artifact was ``np.savez_compressed`` of the coded arrays, 1.37-1.50x
larger than the xz number.  This script removes the ambiguity by writing **the
same container** (``container.write_package``, one xz stream + self-describing
header) for three methods and comparing the files themselves:

    raw      the incumbent layout: float32 payloads, uint8 mask (xz'd by the
             container)                                    -- the pure-xz baseline
    naive8   everything quantised to int8 with one global scale per field, hd
             to int16 (its range forbids 8 bits)            -- the "just quantise"
    ours     the configuration selected by ``run_select_rule.py``

For each (dataset, method, split) a *fresh process* reads and decodes the file,
so the peak RSS figure is the decoder's real footprint, not a residue of the
encoder.  Reported per cell: file bytes, read (decompress+parse) time, decode
(codes -> fields) time, peak RSS, and the MAE obtained from the decoded fields.

The candidate-index segment is measured the same way: an int32 container versus
the 6-bit TOD-offset container, so "the whole chain" is two real files.

Usage:
    python run_container_bench.py [PEMS04 ...]          # run everything
    python run_container_bench.py --one DS METHOD SPLIT  # internal
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import container as PKG                                        # noqa: E402
from run_law_config import EXP, HORIZON, evaluate, load_split  # noqa: E402

OUT = HERE / "container_bench.json"
BENCH = HERE / "packages" / "bench"
FROZEN = HERE / "frozen_rule_v1.json"

METHODS = {
    "raw": {"T": {"codec": "raw"}, "E": {"codec": "raw"},
            "fd": {"codec": "raw"}, "hd": {"codec": "raw"},
            "B": {"codec": "raw"}},
    "naive8": {"T": {"codec": "qint", "bits": 8, "global": True},
               "E": {"codec": "qint", "bits": 8, "global": True},
               "fd": {"codec": "qint", "bits": 8, "global": True},
               "hd": {"codec": "qint", "bits": 16, "global": True},
               "B": {"codec": "raw"}},
}


def ours_plan(ds, variant="min"):
    """The frozen rule's choice.

    ``min``  drops every field whose 0-bit rung passes  (dead state removed)
    ``keep`` keeps those fields at their coarsest live rung (codec only)
    """
    if FROZEN.exists():
        doc = json.loads(FROZEN.read_text("utf-8"))
        sel = doc["selection"].get(ds)
        if sel:
            return sel["plan_keep" if variant == "keep" else "plan"]
    return {"T": {"codec": "qint", "bits": 8}, "E": {"codec": "sparse"},
            "fd": {"codec": "log", "bits": 4}, "hd": {"codec": "qint", "bits": 16}}


class Ctx:
    def __init__(self, ds):
        from emulate_and_build_residual_oracle import FrozenPipeline
        from utils.datasets_utils import file_path_dict, load_raw_data

        ds_dir = ROOT / "datasets" / ds
        meta = np.load(ds_dir / f"{ds}_12to12_index_scaler.npz")
        self.idx = {s: np.asarray(meta[f"idx_{s}"], dtype=np.int64)
                    for s in ("val", "test")}
        self.s_mean = float(np.asarray(meta["scaler_mean"]).item())
        self.s_std = float(np.asarray(meta["scaler_std"]).item())
        raw_all, _ = load_raw_data(file_path_dict[ds])
        self.raw = np.asarray(raw_all[..., 0], dtype=np.float32)
        self.bb = np.load(ds_dir / f"{ds}_12to12_backbone_predictions_joint3.npz")
        self.pipe = FrozenPipeline(ROOT / "EXP_experiments/TLAST_improved" / ds
                                  / EXP[ds], HORIZON, self.raw.shape[1])
        self.pipe.set_codec(None, None, quantize=False)

    def truth(self, split):
        idx = self.idx[split]
        y_raw = self.raw[idx[:, None] + np.arange(1, HORIZON + 1)[None], :]
        return (np.asarray(self.bb[f"backbone_pred_{split}"], dtype=np.float32),
                ((y_raw - self.s_mean) / self.s_std).astype(np.float32),
                y_raw != 0)


# --------------------------------------------------------------------- build --
def build_cache(ds):
    cache = np.load(ROOT / "datasets" / ds
                    / f"{ds}_12to12_analog_k8_exact_mean.npz", allow_pickle=False)
    plans = dict(METHODS, ours_min=ours_plan(ds, "min"),
                 ours_full=ours_plan(ds, "keep"))
    sizes = {}
    for method, plan in plans.items():
        for split in ("val", "test"):
            fields = load_split(cache, split)
            blobs, layout = PKG.encode_fields(fields, plan)
            path = BENCH / f"{ds}__{method}__{split}.fcz"
            size = PKG.write_package(path, blobs, dict(
                kind="retrieval_cache", dataset=ds, split=split, method=method,
                plan=plan, layout=layout))
            sizes[f"{method}/{split}"] = size / 1e6
            del fields, blobs
    cache.close()
    return sizes


def build_index(ds):
    import run_index_codec_audit as R

    paths = R.dataset_paths(ds)
    with np.load(paths["index_scaler"], allow_pickle=False) as z:
        idx_train = np.asarray(z["idx_train"], dtype=np.int64)
        idx_val = np.asarray(z["idx_val"], dtype=np.int64)
        idx_test = np.asarray(z["idx_test"], dtype=np.int64)
    with np.load(paths["time"], allow_pickle=False) as z:
        time_data = np.asarray(z["time"])
    _, groups = R.build_tod_groups(time_data, idx_train)
    gmax = max(g.size for g in groups.values())
    sentinel = gmax
    bits = int(np.ceil(np.log2(gmax + 1)))
    sizes = {}
    with np.load(paths["indices"], allow_pickle=False) as z:
        for split, qidx in (("val", idx_val), ("test", idx_test)):
            key = f"candidate_index_{split}"
            arr = np.asarray(z[key])
            arr2 = arr[..., 0] if arr.ndim == 4 else arr
            qtod = np.asarray(time_data[qidx, 0], dtype=np.int64)
            # raw int32, same container
            blobs = {"codes": np.ascontiguousarray(arr2)}
            size = PKG.write_package(
                BENCH / f"{ds}__index_raw__{split}.fcz", blobs,
                dict(kind="candidate_index_raw", dataset=ds, split=split,
                     layout=dict(kind="raw_index", shape=list(arr2.shape))))
            sizes[f"raw/{split}"] = size / 1e6
            # 6-bit TOD offsets
            blobs, layout = PKG.encode_index(arr2, groups, bits, sentinel, qtod)
            size = PKG.write_package(
                BENCH / f"{ds}__index_sixbit__{split}.fcz", blobs,
                dict(kind="candidate_index", dataset=ds, split=split,
                     layout=layout))
            sizes[f"sixbit/{split}"] = size / 1e6
            del arr, arr2
    return sizes, bits


# ------------------------------------------------------------------ one cell --
def run_one(ds, method, split):
    """Fresh process: read + decode + forward, reporting peak RSS."""
    import psutil
    proc = psutil.Process()
    ctx = Ctx(ds)
    if method == "harness":
        # same imports, same tensors, no package: the peak RSS floor
        P, y, v = ctx.truth(split)
        _ = evaluate(ctx.pipe, P, y, v, ctx.s_std,
                     {"T": np.zeros_like(y), "E": np.zeros_like(y),
                      "fd": np.zeros_like(y), "hd": np.zeros((y.shape[0], y.shape[2]),
                                                            np.float32),
                      "B": np.zeros(y.shape + (1,), np.float32)})
        peak = getattr(proc.memory_info(), "peak_wset", 0) / 1e6
        print(json.dumps(dict(dataset=ds, method=method, split=split, size_MB=0.0,
                              read_ms=0.0, decode_ms=0.0, forward_ms=0.0,
                              RSS_base_MB=proc.memory_info().rss / 1e6,
                              peak_MB=peak, peak_delta_MB=0.0, MAE=None)))
        return
    base = proc.memory_info().rss / 1e6
    t0 = time.perf_counter()
    blobs, header = PKG.read_package(BENCH / f"{ds}__{method}__{split}.fcz")
    t1 = time.perf_counter()
    fields = PKG.decode_fields(blobs, header["layout"])
    t2 = time.perf_counter()
    P, y, v = ctx.truth(split)
    mae = evaluate(ctx.pipe, P, y, v, ctx.s_std, fields)
    t3 = time.perf_counter()
    peak = getattr(proc.memory_info(), "peak_wset", None)
    peak = peak / 1e6 if peak else proc.memory_info().rss / 1e6
    print(json.dumps(dict(
        dataset=ds, method=method, split=split,
        size_MB=(BENCH / f"{ds}__{method}__{split}.fcz").stat().st_size / 1e6,
        read_ms=1e3 * (t1 - t0), decode_ms=1e3 * (t2 - t1),
        forward_ms=1e3 * (t3 - t2), RSS_base_MB=base, peak_MB=peak,
        peak_delta_MB=peak - base, MAE=mae)))


def run_index_one(ds, method, split):
    import psutil
    proc = psutil.Process()
    base = proc.memory_info().rss / 1e6
    t0 = time.perf_counter()
    blobs, header = PKG.read_package(BENCH / f"{ds}__index_{method}__{split}.fcz")
    t1 = time.perf_counter()
    if method == "raw":
        out = np.asarray(blobs["codes"]).astype(np.int32)[..., None]
    else:
        out = PKG.decode_index(blobs, header["layout"])
    t2 = time.perf_counter()
    peak = getattr(proc.memory_info(), "peak_wset", None)
    peak = peak / 1e6 if peak else proc.memory_info().rss / 1e6
    print(json.dumps(dict(
        dataset=ds, method=method, split=split, shape=list(out.shape),
        size_MB=(BENCH / f"{ds}__index_{method}__{split}.fcz").stat().st_size / 1e6,
        read_ms=1e3 * (t1 - t0), decode_ms=1e3 * (t2 - t1),
        RSS_base_MB=base, peak_MB=peak, peak_delta_MB=peak - base)))


# -------------------------------------------------------------------- driver --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=3, metavar=("DS", "METHOD", "SPLIT"))
    ap.add_argument("--index-one", nargs=3, metavar=("DS", "METHOD", "SPLIT"))
    ap.add_argument("datasets", nargs="*", default=[])
    a = ap.parse_args()
    if a.one:
        run_one(*a.one)
        return
    if a.index_one:
        run_index_one(*a.index_one)
        return

    datasets = a.datasets or list(EXP)
    res = {}
    for ds in datasets:
        print(f"\n=== {ds} ===", flush=True)
        t0 = time.perf_counter()
        cache_sizes = build_cache(ds)
        idx_sizes, idx_bits = build_index(ds)
        print(f"  built in {time.perf_counter()-t0:.0f}s "
              f"(index {idx_bits} bit)", flush=True)
        cells = {}
        harness = {}
        for split in ("val", "test"):
            out = subprocess.run(
                [sys.executable, str(Path(__file__)), "--one", ds, "harness", split],
                capture_output=True, text=True, cwd=str(ROOT))
            if out.returncode == 0:
                harness[split] = json.loads(out.stdout.strip().splitlines()[-1])
        for method in ("raw", "naive8", "ours_full", "ours_min"):
            for split in ("val", "test"):
                out = subprocess.run(
                    [sys.executable, str(Path(__file__)), "--one", ds, method, split],
                    capture_output=True, text=True, cwd=str(ROOT))
                if out.returncode != 0:
                    print(out.stderr[-600:])
                    continue
                cells[f"{method}/{split}"] = json.loads(out.stdout.strip().splitlines()[-1])
        idx_cells = {}
        for method in ("raw", "sixbit"):
            for split in ("val", "test"):
                out = subprocess.run(
                    [sys.executable, str(Path(__file__)), "--index-one", ds,
                     method, split], capture_output=True, text=True, cwd=str(ROOT))
                if out.returncode != 0:
                    print(out.stderr[-600:])
                    continue
                idx_cells[f"{method}/{split}"] = json.loads(
                    out.stdout.strip().splitlines()[-1])
        res[ds] = dict(cache_sizes_MB=cache_sizes, index_sizes_MB=idx_sizes,
                       index_bits=idx_bits, harness=harness,
                       cache=cells, index=idx_cells)
        for label, d in (("cache", cells), ("index", idx_cells)):
            print(f"  -- {label}")
            print("  %-8s %5s %9s %9s %9s %9s %9s"
                  % ("method", "split", "MB", "read_ms", "dec_ms", "peak_delta", "MAE/delta"))
            for k, c in d.items():
                method, split = k.split("/")
                mae = c.get("MAE")
                print("  %-8s %5s %9.2f %9.1f %9.1f %9.1f %9s"
                      % (method, split, c["size_MB"], c["read_ms"],
                         c["decode_ms"], c["peak_delta_MB"],
                         f"{mae:.5f}" if mae else "-"))
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
