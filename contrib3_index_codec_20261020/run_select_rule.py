"""Phase B -- turn the measurement into a decision, and freeze it.

Until now the configuration was typed on the command line (``--t-mode i8
--fd-mode log4``).  That is a hand-crafted setting, not a rule: nothing in the
code could reject a bad choice or justify a good one.  This script makes the
choice itself, on validation only, and writes the outcome to a file that is
hashed so a later run cannot silently drift.

The rule (single knob: the MAE gate)
------------------------------------
1. Walk each field's ladder from coarse to fine, measure val MAE, and take the
   **coarsest rung that still clears ``GATE``**.  ``off`` (drop the field) is the
   last rung on purpose: if dropping a field passes, the field is not payload
   and its "knee" must be reported as *unused*, not as a compression result.
2. Assemble the per-field knees into one joint configuration and measure it.
   Individual knees do **not** compose -- on PEMS03 T=int8 (+0.0019), hd=int4
   (-0.0003) and fd=off (+0.0013) each pass, yet the three together cost
   +0.0082, because fd and hd enter the same confidence branch.  So the joint
   configuration is what gets validated.
3. If the joint configuration misses the gate, step fields back up in a
   pre-registered order (fd -> hd -> T -> E, one rung each, round robin) until
   it passes.  The order is fixed before any test number is looked at.
4. Write ``frozen_rule_v1.json`` with the ladders, the gate, the backoff order,
   the per-dataset plan and a SHA-256 over the rule itself.

Usage:
    python run_select_rule.py                 # freeze on validation
    python run_select_rule.py --confirm       # one-shot test, refuses to re-run
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import container as PKG                                        # noqa: E402
from run_law_config import EXP, HORIZON, evaluate, load_split  # noqa: E402

FROZEN = HERE / "frozen_rule_v1.json"
CONFIRM = HERE / "frozen_rule_v1_confirm.json"
GATE = 0.01

# Ordered from 0 bits upward.  ``off`` is the *coarsest* rung, not a fallback:
# if a field survives being dropped, the rule must drop it, because 0 bits is
# cheaper than 2 and still inside the gate.  v1 of this file listed ``off`` last,
# which made the rule select log2 for a field whose 0-bit rung also passed; that
# is a bug in the rule, not a property of the data, and it is recorded here.
LADDERS = {
    "T": [{"codec": "off"}, {"codec": "qint", "bits": 4},
          {"codec": "qint", "bits": 6}, {"codec": "qint", "bits": 8},
          {"codec": "f16"}],
    "E": [{"codec": "off"}, {"codec": "qint", "bits": 8},
          {"codec": "sparse"}],
    "fd": [{"codec": "off"}, {"codec": "log", "bits": 2},
           {"codec": "log", "bits": 4}, {"codec": "log", "bits": 6},
           {"codec": "log", "bits": 8}, {"codec": "f16"}],
    "hd": [{"codec": "off"}, {"codec": "qint", "bits": 2},
           {"codec": "qint", "bits": 4}, {"codec": "qint", "bits": 6},
           {"codec": "qint", "bits": 8}, {"codec": "qint", "bits": 16}],
}
BACKOFF = ["fd", "hd", "T", "E"]

# Measuring one field's ladder must leave the other four fields untouched,
# otherwise "delta at this rung" would silently include the cost of zeroing them.
UNTOUCHED = {"T": {"codec": "raw"}, "E": {"codec": "raw"},
             "fd": {"codec": "raw"}, "hd": {"codec": "raw"},
             "B": {"codec": "raw"}}


def label(spec):
    c = spec["codec"]
    return "off" if c == "off" else (f"{c}{spec['bits']}" if "bits" in spec else c)


def rule_digest():
    canon = json.dumps({"gate": GATE, "ladders": LADDERS, "backoff": BACKOFF},
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


class Data:
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
        self.bb = np.load(ds_dir / f"{ds}_12to12_backbone_predictions_joint3.npz")
        self.pipe = FrozenPipeline(ROOT / "EXP_experiments/TLAST_improved" / ds
                                  / EXP[ds], HORIZON, self.raw.shape[1])
        self.pipe.set_codec(None, None, quantize=False)
        self.cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                             allow_pickle=False)
        self.fields = {s: load_split(self.cache, s) for s in ("val", "test")}
        self.truth = {}
        for s in ("val", "test"):
            idx = self.idx[s]
            y_raw = self.raw[idx[:, None] + np.arange(1, HORIZON + 1)[None], :]
            self.truth[s] = dict(
                P=np.asarray(self.bb[f"backbone_pred_{s}"], dtype=np.float32),
                y=((y_raw - self.s_mean) / self.s_std).astype(np.float32),
                v=y_raw != 0)

    def mae(self, split, plan, with_c0=False):
        blobs, layout = PKG.encode_fields(self.fields[split], plan)
        fields = PKG.decode_fields(blobs, layout)
        t = self.truth[split]
        m = evaluate(self.pipe, t["P"], t["y"], t["v"], self.s_std, fields)
        if with_c0:
            return m, evaluate(self.pipe, t["P"], t["y"], t["v"], self.s_std,
                               self.fields[split])
        return m

    def packed_MB(self, split, plan):
        """Exact size of the coded blobs, before the generic codec.

        The ladder is decided by MAE; measuring a full xz package for every
        rung would cost ~20 compressions of a 239 MB field on PEMS07 and add no
        decision-relevant information.  The shipped size is measured on the
        joint configuration only.
        """
        blobs, _ = PKG.encode_fields(self.fields[split], plan)
        return sum(v.nbytes for v in blobs.values()) / 1e6

    def package_MB(self, split, plan, tier="rule"):
        blobs, layout = PKG.encode_fields(self.fields[split], plan)
        path = HERE / "packages" / "select" / f"{self.ds}__{tier}__{split}.fcz"
        size = PKG.write_package(path, blobs, dict(
            kind="retrieval_cache", dataset=self.ds, split=split, tier=tier,
            plan=plan, layout=layout))
        return size / 1e6


def run_dataset(ds, verbose=True):
    d = Data(ds)
    # C0 = untouched fields
    base_val = evaluate(d.pipe, d.truth["val"]["P"], d.truth["val"]["y"],
                        d.truth["val"]["v"], d.s_std, d.fields["val"])

    ladder, knee = {}, {}
    if verbose:
        print(f"\n=== {ds}  C0(val) {base_val:.5f}  gate {GATE} ===")
        print(f"{'field':6s} {'rung':8s} {'val_delta':>10s} {'MB':>7s} {'pass':>5s}")
    for name in ("T", "E", "fd", "hd"):
        rows, chosen = [], None
        for spec in LADDERS[name]:
            plan = dict(UNTOUCHED, **{name: spec})       # only this field moves
            delta = d.mae("val", plan) - base_val
            # exact coded bytes of this field alone, before xz
            mb = d.packed_MB("val", {name: spec})
            ok = delta <= GATE
            rows.append(dict(rung=label(spec), delta=delta, MB=mb, pass_=bool(ok)))
            if ok and chosen is None:
                chosen = spec
            if verbose:
                print(f"{name:6s} {label(spec):8s} {delta:+10.5f} {mb:7.3f} "
                      f"{'yes' if ok else 'NO':>5s}")
        ladder[name] = rows
        knee[name] = chosen or LADDERS[name][-1]
        if verbose:
            print(f"       -> knee {label(knee[name])}")

    plan = dict(knee)
    joint0 = d.mae("val", plan) - base_val
    steps, cursor, guard = [], 0, 0
    while joint0 > GATE and guard < 24:
        guard += 1
        name = BACKOFF[cursor % len(BACKOFF)]
        cursor += 1
        order = [label(s) for s in LADDERS[name]]
        cur = label(plan[name])
        i = order.index(cur)
        if i >= len(LADDERS[name]) - 1:
            continue                       # already at the finest rung
        plan[name] = LADDERS[name][i + 1]
        steps.append(f"{name}:{cur}->{label(plan[name])}")
        joint0 = d.mae("val", plan) - base_val
    if verbose:
        print(f"  joint {joint0:+.5f} after {len(steps)} backoff step(s) "
              f"{steps or '-'}")

    # Second reading, same rule, drops forbidden: every field that the ladder
    # would delete is kept at its finest *cheap* rung.  This is the number the
    # codec itself can claim; the drop-dead plan also earns the dead state.
    keep = {name: (LADDERS[name][1] if spec["codec"] == "off" else spec)
            for name, spec in plan.items()}
    keep_delta = d.mae("val", keep) - base_val
    cursor2, guard2 = 0, 0
    while keep_delta > GATE and guard2 < 24:
        guard2 += 1
        name = BACKOFF[cursor2 % len(BACKOFF)]
        cursor2 += 1
        order = [label(s) for s in LADDERS[name]]
        i = order.index(label(keep[name]))
        if i >= len(LADDERS[name]) - 1:
            continue
        keep[name] = LADDERS[name][i + 1]
        keep_delta = d.mae("val", keep) - base_val
    if verbose:
        print(f"  keep-all {keep_delta:+.5f}  "
              f"{ {k: label(v) for k, v in keep.items()} }")

    out = dict(C0_val=base_val, gate=GATE, ladder=ladder,
               knee={k: label(v) for k, v in knee.items()},
               joint_val_delta=joint0, backoff_steps=steps,
               plan={k: v for k, v in plan.items()},
               plan_keep={k: v for k, v in keep.items()},
               keep_val_delta=keep_delta,
               package_MB={s: d.package_MB(s, plan) for s in ("val", "test")},
               package_keep_MB={s: d.package_MB(s, keep, tier="keep")
                                for s in ("val", "test")})
    d.cache.close()
    del d
    return out


def freeze(datasets):
    res = {}
    for ds in datasets:
        t0 = time.perf_counter()
        res[ds] = run_dataset(ds)
        print(f"  [{ds}] {time.perf_counter()-t0:.0f}s", flush=True)
    doc = dict(rule_id="fcl-rule-v1", gate=GATE, ladders=LADDERS,
               backoff_order=BACKOFF, rule_sha256=rule_digest(),
               frozen_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               selection=res, test_used=False)
    FROZEN.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\nfrozen -> {FROZEN}  sha {doc['rule_sha256'][:16]}")
    tot_v = sum(v["package_MB"]["val"] for v in res.values())
    tot_t = sum(v["package_MB"]["test"] for v in res.values())
    print(f"cache package bytes (val+test, 4 ds): {tot_v + tot_t:.2f} MB")


def confirm():
    if CONFIRM.exists():
        raise SystemExit(f"{CONFIRM} exists: test was already used. "
                         "Delete it deliberately if you really mean to re-run.")
    doc = json.loads(FROZEN.read_text("utf-8"))
    if doc.get("test_used"):
        raise SystemExit("frozen rule is marked test_used")
    rows = {}
    print("%-8s %10s %10s %8s" % ("ds", "val_delta", "test_delta", "MB_test"))
    for ds, sel in doc["selection"].items():
        d = Data(ds)
        base = evaluate(d.pipe, d.truth["test"]["P"], d.truth["test"]["y"],
                        d.truth["test"]["v"], d.s_std, d.fields["test"])
        mae = d.mae("test", sel["plan"])
        rows[ds] = dict(val_delta=sel["joint_val_delta"], test_delta=mae - base,
                        MB=d.package_MB("test", sel["plan"], tier="frozen"))
        print("%-8s %+10.5f %+10.5f %8.2f"
              % (ds, rows[ds]["val_delta"], rows[ds]["test_delta"], rows[ds]["MB"]))
        d.cache.close()
        del d
    doc["test_used"] = True
    FROZEN.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    CONFIRM.write_text(json.dumps(dict(
        rule_sha256=doc["rule_sha256"],
        confirmed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rows=rows), indent=1), encoding="utf-8")
    print(f"\nconfirm -> {CONFIRM}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("datasets", nargs="*", default=[])
    a = ap.parse_args()
    if a.confirm:
        confirm()
    else:
        freeze(a.datasets or list(EXP))


if __name__ == "__main__":
    main()
