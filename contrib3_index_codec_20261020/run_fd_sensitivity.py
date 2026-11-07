"""Is fd's precision-insensitivity a *law* or is fd simply unused?

The three log-domain codes (8/6/4 bits) all landed on the same +0.00105 MAE on
PEMS04, which is suspiciously clean.  Two readings are possible:

  (a) the codec is fine because fd only carries ordinal / saturation
      information for the confidence head -- a real statement about *where* the
      bit budget can go;
  (b) the frozen checkpoint barely uses fd, so *any* fd code (even zero) would
      pass -- in which case "fd tolerates 4 bits" is an artefact of this
      checkpoint and must not be sold as a law.

This probe separates the two by pushing fd through a ladder of increasingly
brutal degradations and measuring MAE.  Degradations used:

    f16            current codec
    log8/6/4/2     the proposals
    constant       every value -> median of the non-zero values
    shuffled       marginal kept, sample pairing destroyed
    zero           fd removed entirely

`shuffled` is the informative one: if it is as harmless as `constant`, then
only the *spread* of fd matters, not its value per sample.  Validation only;
no training, no test peeking.

Usage:  python run_fd_sensitivity.py PEMS04 PEMS08
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import codec_fields as CF                                     # noqa: E402
from run_law_config import EXP, HORIZON, evaluate, load_split  # noqa: E402
from emulate_and_build_residual_oracle import FrozenPipeline   # noqa: E402

OUT = HERE / "fd_sensitivity.json"


def main():
    datasets = sys.argv[1:] or ["PEMS04", "PEMS08"]
    res = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    for ds in datasets:
        ds_dir = ROOT / "datasets" / ds
        exp_dir = ROOT / "EXP_experiments/TLAST_improved" / ds / EXP[ds]
        meta = np.load(ds_dir / f"{ds}_12to12_index_scaler.npz")
        idx_val = np.asarray(meta["idx_val"], dtype=np.int64)
        s_mean = float(np.asarray(meta["scaler_mean"]).item())
        s_std = float(np.asarray(meta["scaler_std"]).item())

        from utils.datasets_utils import file_path_dict, load_raw_data
        raw_all, _ = load_raw_data(file_path_dict[ds])
        raw = np.asarray(raw_all[..., 0], dtype=np.float32)
        n_nodes = raw.shape[1]

        bb = np.load(ds_dir / f"{ds}_12to12_backbone_predictions_joint3.npz")
        pipe = FrozenPipeline(exp_dir, HORIZON, n_nodes)
        pipe.set_codec(None, None, quantize=False)

        cache = np.load(ds_dir / f"{ds}_12to12_analog_k8_exact_mean.npz",
                        allow_pickle=False)
        val = load_split(cache, "val")
        P = np.asarray(bb["backbone_pred_val"], dtype=np.float32)
        y_raw = raw[idx_val[:, None] + np.arange(1, HORIZON + 1)[None], :]
        y = ((y_raw - s_mean) / s_std).astype(np.float32)
        v = y_raw != 0
        c0 = evaluate(pipe, P, y, v, s_std, val)

        fd = val["fd"]
        med = float(np.median(fd[fd > 0]))
        shuf = np.random.default_rng(0).permutation(
            fd.reshape(-1)).reshape(fd.shape).astype(np.float32)

        ladder = {"f16": CF.decode_fd(CF.encode_fd(fd))}
        for bits in (8, 6, 4, 2):
            c, m = CF.encode_fd_log(fd, bits)
            ladder[f"log{bits}"] = CF.decode_fd_log(c, m, bits)
        ladder["constant"] = np.full_like(fd, np.float32(med))
        ladder["shuffled"] = shuf
        ladder["zero"] = np.zeros_like(fd)

        rows = {}
        print(f"--- {ds} val (C0 {c0:.5f}, fd median {med:.2f}) ---")
        print(f"{'variant':10s} {'MAE':>9s} {'delta':>9s} {'z_p99':>8s}")
        for name, arr in ladder.items():
            alt = dict(val)
            alt["fd"] = arr
            mae = evaluate(pipe, P, y, v, s_std, alt)
            dz = np.abs(arr / s_std - fd / s_std)
            rows[name] = dict(MAE=mae, delta=mae - c0,
                              z_p99=float(np.percentile(dz, 99)))
            print(f"{name:10s} {mae:9.5f} {mae-c0:+9.5f} "
                  f"{rows[name]['z_p99']:8.4f}", flush=True)
            del alt
        res[ds] = dict(C0=c0, fd_median=med, ladder=rows)
        cache.close()
        del pipe, P, y, v, val, ladder
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
