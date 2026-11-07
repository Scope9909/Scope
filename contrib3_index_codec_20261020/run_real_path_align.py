"""Real-path alignment for contribution 3.

Why this exists
---------------
Every Delta-MAE number produced so far came from
``emulate_and_build_residual_oracle.FrozenPipeline``: a hand written numpy
pointwise re-implementation whose acceptance test was run under
``--reranker-uniform``.  The four checkpoints we report all carry
``use_analog_candidate_reranker = 1``, so the emulator and the real model are
not the same function.

This script closes that gap the way the audit asked for: first reproduce the
recorded baseline on the *real* full forward path, then replay the encoded
state on that same path.

STEP 0  replay the untouched cache on the real path and compare against the
        numbers recorded in the training logs.  If this does not reproduce,
        nothing downstream can be trusted.

Usage
    python run_real_path_align.py step0
    python step1 later, once the container fields have been written back.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
BENCH = ROOT / "benchmark_runtime_code_domain.py"
OUT = HERE / "real_path_align.json"

EXP = {
    "PEMS03": "TLAST_improved_PEMS03_12to12_exppaper_pems03_s1_final_joint",
    "PEMS04": "TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3",
    "PEMS07": "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint",
    "PEMS08": "TLAST_improved_PEMS08_12to12_exppaper_pems08_s1_final_joint",
}

# recorded in the training logs of the very same experiment directories
RECORDED = {
    "PEMS03": {"val": 13.1907, "test": 14.1924},
    "PEMS04": {"val": 17.8170, "test": 17.6296},
    "PEMS07": {"val": 18.0889, "test": 18.8438},
    "PEMS08": {"val": 12.9793, "test": 12.9798},
}

# the emulator's own replay of the same untouched cache, for the gap column
EMULATED = {
    "PEMS03": {"val": 13.19873, "test": 14.20526},
    "PEMS04": {"val": 17.85202, "test": 17.67307},
    "PEMS07": {"val": 18.22224, "test": 18.95955},
    "PEMS08": {"val": 13.24612, "test": 13.25865},
}

DS_ORDER = ["PEMS08", "PEMS04", "PEMS03", "PEMS07"]


def run_eval(ds, split, retrieval_cache=None, tag="raw"):
    """One real GPU eval on the untouched full forward path."""
    exp_dir = ROOT / "EXP_experiments/TLAST_improved" / ds / EXP[ds]
    out = HERE / f"_align_{ds}_{tag}_{split}.json"
    cmd = [
        str(PY), str(BENCH),
        "--mode", "baseline",
        "--exp-dir", str(exp_dir),
        "--split", split,
        "--output", str(out),
    ]
    if retrieval_cache is not None:
        cmd += ["--retrieval-cache", str(retrieval_cache)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if not out.exists():
        return None, (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
    return json.loads(out.read_text(encoding="utf-8")), None


def step0():
    res = {}
    print("STEP 0 -- reproduce the recorded baseline on the real full path")
    print(f"{'ds':8s} {'split':6s} {'real':>10s} {'logged':>10s} "
          f"{'replay-gap':>11s} {'emulated':>10s} {'emu-gap':>9s} {'verdict':>8s}")
    for ds in DS_ORDER:
        res[ds] = {}
        for split in ("val", "test"):
            doc, err = run_eval(ds, split)
            if doc is None:
                print(f"{ds:8s} {split:6s} FAILED\n{err}")
                res[ds][split] = {"error": err}
                continue
            real = float(doc["metrics"]["mae"])
            logged = RECORDED[ds][split]
            emu = EMULATED[ds][split]
            gap = real - logged
            ok = abs(gap) < 5e-3
            res[ds][split] = dict(
                real_MAE=real, logged_MAE=logged, replay_gap=gap,
                emulated_MAE=emu, emulator_gap=emu - real,
                query_count=doc["query_count"], reproduced=bool(ok),
                reranker=doc.get("reranker"), scope=doc.get("scope"))
            print(f"{ds:8s} {split:6s} {real:10.5f} {logged:10.4f} "
                  f"{gap:+11.5f} {emu:10.5f} {emu - real:+9.5f} "
                  f"{'OK' if ok else 'MISMATCH':>8s}", flush=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nJSON -> {OUT}")
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "step0"
    if which == "step0":
        step0()
    else:
        raise SystemExit(f"unknown step {which}")
