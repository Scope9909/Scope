"""Phase A -- prove the package decodes on its own.

The old measurement path was ``decode(cfg, ref)``: the fired mask came from
``ref["B"]``, the E shape from ``ref["E"]``, the fd shape from ``ref["fd"]``.
So "the coded artifact" was never actually decodable alone.  This script closes
that hole and demonstrates it in the strongest available way:

  stage 1  build    -- encode val/test, write ``packages/<ds>__<tier>__<split>.fcz``
  stage 2  decode   -- **separate process**: reads only the .fcz + model weights
                       + ground truth.  ``numpy.load`` is patched to raise if any
                       retrieval-cache file is touched, so a hidden dependency
                       would crash instead of silently passing.

Acceptance criteria (both printed):
  A1  decoded fields == re-encoded fields, elementwise (container is lossless
      w.r.t. the codec -- no new approximation is smuggled in by the wrapper)
  A2  MAE from the standalone decode == the MAE recorded by the in-memory run
      (|diff| <= 1e-9), for the recommended and the extreme tier

Usage:
    python run_independent_decode.py build
    python run_independent_decode.py verify
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import container as PKG                                        # noqa: E402
import codec_fields as CF                                      # noqa: E402
from run_law_config import EXP, HORIZON, evaluate, load_split  # noqa: E402

PACKDIR = HERE / "packages"
MANIFEST = HERE / "independent_decode.json"

TIERS = {
    "rec": {"T": {"codec": "qint", "bits": 8},      # knee on 3/4 datasets
            "E": {"codec": "sparse"},
            "fd": {"codec": "log", "bits": 4},
            "hd": {"codec": "qint", "bits": 16}},
    "ext": {"T": {"codec": "qint", "bits": 8},
            "E": {"codec": "sparse"},
            "fd": {"codec": "off"},
            "hd": {"codec": "qint", "bits": 4}},
}
REFERENCE = {"rec": "law_config_i8_log4.json", "ext": "law_config_i8_zero.json"}


def reference_fields(fields, plan):
    """Independent re-implementation of the same plan, via ``codec_fields``.

    A1 asks whether the *container* is lossless with respect to the codec, i.e.
    whether packing at exactly N bits reproduces what the old, unpacked codec
    path produced.  Comparing against the original (unquantised) field would
    only restate that the codec is lossy, which is known.
    """
    S, H, N = fields["T"].shape
    fired = fields["B"][..., 0] > 0.5
    out = {}
    spec = plan["T"]
    if spec["codec"] == "qint":
        out["T"] = CF.decode_hd(*CF.encode_hd(fields["T"], bits=spec["bits"]))
    elif spec["codec"] == "f16":
        out["T"] = fields["T"].astype(np.float16).astype(np.float32)
    else:
        out["T"] = np.zeros((S, H, N), np.float32)
    spec = plan["E"]
    if spec["codec"] == "sparse":
        out["E"] = CF.decode_e_sparse(CF.encode_e_sparse(fields["E"], fired),
                                      fired, fields["E"].shape)
    elif spec["codec"] == "qint":
        out["E"] = CF.decode_hd(*CF.encode_hd(fields["E"], bits=spec["bits"]))
    else:
        out["E"] = np.zeros((S, H, N), np.float32)
    spec = plan["fd"]
    if spec["codec"] == "log":
        out["fd"] = CF.decode_fd_log(*CF.encode_fd_log(fields["fd"], spec["bits"]),
                                     spec["bits"])
    elif spec["codec"] == "f16":
        out["fd"] = CF.decode_fd(CF.encode_fd(fields["fd"]))
    else:
        out["fd"] = np.zeros((S, H, N), np.float32)
    spec = plan["hd"]
    if spec["codec"] == "qint":
        out["hd"] = CF.decode_hd(*CF.encode_hd(fields["hd"], bits=spec["bits"]))
    elif spec["codec"] == "f16":
        out["hd"] = fields["hd"].astype(np.float16).astype(np.float32)
    else:
        out["hd"] = np.zeros((S, N), np.float32)
    out["B"] = fired.astype(np.float32)[..., None]
    return out
GATE = 0.01


# ------------------------------------------------------------------ context --
class Ctx:
    """Everything that is *not* the cache: weights, inputs, labels."""

    def __init__(self, ds):
        from emulate_and_build_residual_oracle import FrozenPipeline
        from utils.datasets_utils import file_path_dict, load_raw_data

        self.ds = ds
        ds_dir = ROOT / "datasets" / ds
        meta = np.load(ds_dir / f"{ds}_12to12_index_scaler.npz")
        self.idx = {s: np.asarray(meta[f"idx_{s}"], dtype=np.int64)
                    for s in ("val", "test")}
        self.s_mean = float(np.asarray(meta["scaler_mean"]).item())
        self.s_std = float(np.asarray(meta["scaler_std"]).item())
        raw_all, _ = load_raw_data(file_path_dict[ds])
        self.raw = np.asarray(raw_all[..., 0], dtype=np.float32)
        self.n_nodes = self.raw.shape[1]
        self.bb = np.load(ds_dir / f"{ds}_12to12_backbone_predictions_joint3.npz")
        self.pipe = FrozenPipeline(ROOT / "EXP_experiments/TLAST_improved" / ds
                                  / EXP[ds], HORIZON, self.n_nodes)
        self.pipe.set_codec(None, None, quantize=False)

    def targets(self, split):
        idx = self.idx[split]
        y_raw = self.raw[idx[:, None] + np.arange(1, HORIZON + 1)[None], :]
        y = ((y_raw - self.s_mean) / self.s_std).astype(np.float32)
        P = np.asarray(self.bb[f"backbone_pred_{split}"], dtype=np.float32)
        return P, y, y_raw != 0


# -------------------------------------------------------------------- build --
def build(only=None):
    man = {}
    for ds in (only or list(EXP)):
        ctx = Ctx(ds)
        cache = np.load(ROOT / "datasets" / ds
                        / f"{ds}_12to12_analog_k8_exact_mean.npz",
                        allow_pickle=False)
        print(f"\n=== {ds} ===", flush=True)
        for tier, plan in TIERS.items():
            for split in ("val", "test"):
                fields = load_split(cache, split)
                blobs, layout = PKG.encode_fields(fields, plan)
                path = PACKDIR / f"{ds}__{tier}__{split}.fcz"
                size = PKG.write_package(path, blobs, dict(
                    kind="retrieval_cache", dataset=ds, split=split,
                    tier=tier, plan={k: v for k, v in plan.items()},
                    layout=layout))
                # A1a: decode right here and demand exactness vs the encoder
                back = PKG.package_fields(path)[0]
                want = reference_fields(fields, plan)
                exact = {k: bool(np.array_equal(back[k], want[k]))
                         for k in ("T", "E", "fd", "hd", "B")}
                P, y, v = ctx.targets(split)
                mae = evaluate(ctx.pipe, P, y, v, ctx.s_std, back)
                c0 = evaluate(ctx.pipe, P, y, v, ctx.s_std, fields)
                man.setdefault(ds, {}).setdefault(tier, {})[split] = dict(
                    package=str(path.relative_to(HERE)), size_MB=size / 1e6,
                    exact=exact, C0=c0, MAE=mae, delta=mae - c0,
                    raw_MB=sum(fields[k].nbytes for k in fields) / 1e6)
                print(f"  {tier:3s} {split:4s} pkg {size/1e6:7.2f} MB  "
                      f"exact {all(exact.values())}  C0 {c0:.5f}  "
                      f"MAE {mae:.5f}  d {mae-c0:+.5f}", flush=True)
                del fields, blobs, back
        cache.close()
        del ctx
    MANIFEST.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"\nmanifest -> {MANIFEST}")
    return man


# ------------------------------------------------------------------- decode --
_DECODE_ONLY = False
_real_load = np.load


def _guarded_load(*a, **kw):                                   # pragma: no cover
    first = str(a[0]) if a else str(kw.get("file", ""))
    if _DECODE_ONLY and ("analog_k8_exact_mean" in first
                         or "candidate_indices" in first):
        raise RuntimeError("decode-only mode touched the retrieval cache: "
                           f"{first}")
    return _real_load(*a, **kw)


np.load = _guarded_load


def decode_one(package, ds, split):
    global _DECODE_ONLY
    _DECODE_ONLY = True
    ctx = Ctx(ds)
    fields, header = PKG.package_fields(package)
    P, y, v = ctx.targets(split)
    mae = evaluate(ctx.pipe, P, y, v, ctx.s_std, fields)
    return dict(dataset=ds, split=split, MAE=mae, tier=header["tier"],
                size_MB=Path(package).stat().st_size / 1e6,
                header_bytes=len(json.dumps(header)))


# ------------------------------------------------------------------- verify --
def verify():
    man = json.loads(MANIFEST.read_text("utf-8"))
    ref = {}
    for tier, name in REFERENCE.items():
        d = json.loads((HERE / name).read_text("utf-8"))
        ref[tier] = {ds: {s: d[ds]["splits"][s]["delta"] for s in ("val", "test")}
                     for ds in d if "error" not in d[ds]}
    rows, ok = [], True
    print("\n%-8s %4s %5s %9s %9s %9s %8s %6s"
          % ("ds", "tier", "split", "ref_delta", "dec_delta", "|diff|", "MB", "verdict"))
    for ds, tiers in man.items():
        for tier, splits in tiers.items():
            for split, rec in splits.items():
                pkg = HERE / rec["package"]
                out = subprocess.run(
                    [sys.executable, str(Path(__file__)), "decode",
                     str(pkg), ds, split],
                    capture_output=True, text=True, cwd=str(ROOT))
                if out.returncode != 0:
                    print(f"{ds} {tier} {split}: DECODE FAILED\n{out.stderr[-800:]}")
                    ok = False
                    continue
                got = json.loads(out.stdout.strip().splitlines()[-1])
                delta = got["MAE"] - rec["C0"]
                want = ref[tier][ds][split]
                diff = abs(delta - want)
                good = diff <= 1e-9 and all(rec["exact"].values())
                ok &= good
                rows.append(dict(dataset=ds, tier=tier, split=split,
                                 ref_delta=want, decoded_delta=delta,
                                 abs_diff=diff, size_MB=got["size_MB"],
                                 exact=rec["exact"], pass_=bool(good)))
                print("%-8s %4s %5s %+9.5f %+9.5f %9.1e %8.2f %6s"
                      % (ds, tier, split, want, delta, diff,
                         got["size_MB"], "PASS" if good else "FAIL"))
    (HERE / "independent_decode_verify.json").write_text(
        json.dumps(dict(all_pass=bool(ok), rows=rows), indent=1), encoding="utf-8")
    print(f"\nALL PASS: {ok}")
    return 0 if ok else 1


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    if stage == "build":
        build(sys.argv[2:] or None)
    elif stage == "verify":
        sys.exit(verify())
    elif stage == "decode":
        pkg, ds, split = sys.argv[2], sys.argv[3], sys.argv[4]
        print(json.dumps(decode_one(pkg, ds, split)))
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
