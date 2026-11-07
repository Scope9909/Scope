"""emulate_and_build_residual_oracle.py

Zero-training, validation-only.  Two jobs.

JOB 1 -- deployment-faithful offline emulator
---------------------------------------------
Under ``benchmark_runtime_code_domain.py --mode baseline --reranker-uniform``
the candidate reranker's output projection is zeroed after loading, so its
logits are identically zero, ``weights == uniform`` and

    correction = ((weights - uniform) * candidate_future).sum(dim=1) == 0

(see ``AnalogCandidateHorizonReranker.forward``).  The returned value is then
exactly ``mean_retrieval``, i.e. retrieval cache channel 0 (r8-decoded).  With
the joint3 flags the remaining active stages are

    o1 = P + g      * (T - P)                  # low-rank analog residual
    o2 = o1 + w[h]  * avail * (E - o1)         # exact-history residual
    a  = 0.25 * tanh(q . W + <q, NQF x NF> + b[h])
    o3 = o2 + a     * (T - o2)                 # confidence refinement

where T is the retrieval cache channel 0, P the frozen backbone forecast,
E/avail the exact-continuation cache, and

    q = [hd, fd, exact_avail, signed_gap, |gap|, relative_gap]

with ``gap = T - o2``.  Every one of these operations is pointwise in
(sample, horizon, node): once the reranker is uniform there is no cross-node,
cross-horizon or cross-candidate mixing left.

Consequence: the deployed MAE is exactly reproducible offline, and the oracle
-- for any candidate-selection granularity that keeps the selection
(sample,horizon,node)-local -- is *exactly* computable rather than estimated.

The emulator verifies itself against numbers recorded by real GPU benchmark
runs (canonical plus the whole candrel beta sweep) before anything is built.

JOB 2 -- residual oracle at four selection granularities
-------------------------------------------------------
For a shortlist of K candidates the retrieval cache channel 0 can be set to
the residual of any single candidate.  Because the deployed pipeline is
pointwise, four increasingly fine selections can be evaluated exactly:

    per query            one candidate for the whole sample
    per (query,node)     one candidate per node, shared over horizons
    per (query,horizon)  one candidate per horizon, shared over nodes
    per (q,h,n)          fully element-local

``AnalogCandidateHorizonReranker`` produces per-(batch,candidate,horizon,node)
weights, so the third and fourth granularities are architecturally reachable
reachable by a reranker; the first two are the coarser baselines.

An oracle picks the candidate with the query future ``Y_q`` known, so every
number here is an UPPER BOUND / DIAGNOSTIC, not a model result.  It measures
how much exploitable information sits inside the shortlist.

Only the validation split is ever touched.  ``retrieval_code_train`` and
``retrieval_code_test`` are copied verbatim from the canonical cache.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from build_candidate_relative_residual_cache import (
    build_pos_lookup,
    gather_candidate_future,
    gather_candidate_parent,
)

Y_OFFSETS = np.arange(1, 13, dtype=np.int64)
JOINT3 = Path(
    "EXP_experiments/TLAST_improved/PEMS04/"
    "TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3"
)
GRANULARITIES = ("query", "query_node", "query_horizon", "element")

# Numbers produced by real GPU benchmark runs, used as the emulator's
# acceptance test; keys are retrieval cache file names.
RECORDED = {
    "PEMS04_12to12_analog_k8_exact_mean_latc_pca_r8_fp16.npz": 17.8537288,
    "PEMS04_12to12_analog_k8_exact_mean_candrel_pca_r8_fp16.npz": 17.8185789,
    "PEMS04_12to12_analog_k8_exact_mean_candrel_b15_pca_r8_fp16.npz": 17.8104899,
    "PEMS04_12to12_analog_k8_exact_mean_candrel_b20_pca_r8_fp16.npz": 17.8073615,
    "PEMS04_12to12_analog_k8_exact_mean_candrel_b25_pca_r8_fp16.npz": 17.8087890,
    "PEMS04_12to12_analog_k8_exact_mean_candrel_b30_pca_r8_fp16.npz": 17.8143870,
    "PEMS04_12to12_analog_k8_exact_mean_candrel_b40_pca_r8_fp16.npz": 17.8369200,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=Path("datasets/PEMS04"))
    p.add_argument("--tag", default="PEMS04_12to12")
    p.add_argument("--joint3-exp", type=Path, default=JOINT3)
    p.add_argument(
        "--pca-cache",
        type=Path,
        default=Path(
            "datasets/PEMS04/"
            "PEMS04_12to12_analog_k8_exact_mean_latc_pca_r8_fp16.npz"
        ),
    )
    p.add_argument(
        "--candidate-cache",
        type=Path,
        default=Path(
            "datasets/PEMS04/PEMS04_12to12_analog_k8_candidate_indices.npz"
        ),
    )
    p.add_argument(
        "--backbone-dump",
        type=Path,
        default=Path(
            "datasets/PEMS04/PEMS04_12to12_backbone_predictions_joint3.npz"
        ),
    )
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument(
        "--betas", type=float, nargs="+", default=[1.0, 1.5, 2.0],
        help="Oracle residual transfer strengths to evaluate.",
    )
    p.add_argument(
        "--build-beta", type=float, default=1.0,
        help="Which oracle beta to materialise as caches.",
    )
    p.add_argument(
        "--build-granularities",
        nargs="+",
        choices=GRANULARITIES,
        default=["query_horizon", "element"],
        help="Oracle caches to materialise (requires --output-dir).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for the built oracle caches; omit to evaluate only.",
    )
    p.add_argument(
        "--report", type=Path,
        default=Path("residual_oracle_deployed_diagnostic.json"),
    )
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument(
        "--eval-caches", nargs="*", default=None,
        help="Extra retrieval caches to score, for cross-checking a real run.",
    )
    return p.parse_args()


def decode_codes(code, temporal_mean, temporal_basis):
    """(S,R,N,C) fp16 codes -> (S,H,N,C) float32 retrieval values."""
    return temporal_mean[None] + np.einsum(
        "srnc,nhr->shnc",
        np.asarray(code, dtype=np.float32),
        temporal_basis,
        optimize=True,
    )


class FrozenPipeline:
    """Pointwise emulation of the deployed joint3 validation path."""

    def __init__(self, exp_dir: Path, horizon: int, n_nodes: int):
        with (exp_dir / "config.pkl").open("rb") as handle:
            cfg = pickle.load(handle)
        ck = torch.load(str(exp_dir / "best.pth"), map_location="cpu")

        self.horizon = horizon
        self.n_nodes = n_nodes

        self.blend_max = float(getattr(cfg, "analog_retrieval_max", 1.0))
        hl = ck["analog_lowrank_retrieval_residual.horizon_logits"].numpy()
        nf = ck["analog_lowrank_retrieval_residual.node_factor"].numpy()
        hf = ck["analog_lowrank_retrieval_residual.horizon_factor"].numpy()
        self.g = (self.blend_max * np.tanh(hl[:, None] + hf @ nf.T)).astype(
            np.float32
        )  # (H,N)

        self.exact_max = float(getattr(cfg, "analog_exact_match_max", 1.0))
        self.w = (
            self.exact_max
            * np.tanh(ck["analog_exact_match_residual.horizon_logits"].numpy())
        ).astype(np.float32)  # (H,)

        self.conf_max = float(getattr(cfg, "analog_confidence_max", 0.25))
        pre = "analog_confidence_aware_retrieval_residual."
        self.cw = ck[pre + "quality_weight"].numpy()  # (H,6)
        self.cb = ck[pre + "horizon_bias"].numpy()  # (H,)
        self.cnf = ck[pre + "node_factor"].numpy()  # (N,r)
        self.cnqf = ck[pre + "node_quality_factor"].numpy()  # (H,6,r)

        self._mean = None
        self._basis = None

    def set_codec(self, temporal_mean, temporal_basis, quantize=True):
        """Route T through the cache's r8 codec exactly as the loader does.

        The loader reads ``retrieval_code_*`` (fp16) and reconstructs
        ``temporal_mean + code @ basis``, so whatever a cache stores is
        compressed to rank 8 *per node* before the model sees it.  Modelling
        that step is what makes an oracle cache comparable to a real run: a
        one-hot candidate field is far off the canonical manifold, so r8
        round-trip error is much larger there than for the mean field.
        """
        self._mean = np.asarray(temporal_mean, dtype=np.float32)
        self._basis = np.asarray(temporal_basis, dtype=np.float32)
        self.quantize = quantize

    def codec(self, T):
        if not getattr(self, "quantize", False):
            return T
        code = np.einsum(
            "shnc,nhr->srnc",
            T - self._mean[None],
            self._basis,
            optimize=True,
        ).astype(np.float16)
        return self._mean[None] + np.einsum(
            "srnc,nhr->shnc",
            code.astype(np.float32),
            self._basis,
            optimize=True,
        )

    def forward(self, P, T, E, avail, q0, q1, q2):
        """P,T,E,avail: (S,H,N,C).  q0,q1,q2: (S,H,N) quality scalars."""
        T = self.codec(T)
        o1 = P + self.g[None, :, :, None] * (T - P)
        o2 = o1 + self.w[None, :, None, None] * avail * (E - o1)

        gap = (T - o2)[..., 0]
        signed = gap / (1.0 + np.abs(gap))
        absolute = np.abs(gap) / (1.0 + np.abs(gap))
        rel_raw = gap / np.maximum(np.abs(o2[..., 0]), 1e-3)
        relative = rel_raw / (1.0 + np.abs(rel_raw))

        q = np.stack([q0, q1, q2, signed, absolute, relative], axis=-1)
        logits = (
            np.einsum("shni,hi->shn", q, self.cw)
            + np.einsum("shni,hir,nr->shn", q, self.cnqf, self.cnf)
            + self.cb[None, :, None]
        )
        adjust = (self.conf_max * np.tanh(logits))[..., None]
        return o2 + adjust * (T - o2)


def quality_scalars(cache, scaler_std, horizon):
    """Reproduce the loader's quality channels, then the module's squashing."""
    hd = np.asarray(cache["history_distance_val"], dtype=np.float32)  # (S,N)
    fd = np.asarray(cache["future_dispersion_val"], dtype=np.float32)  # (S,H,N,1)
    ex = np.asarray(cache["exact_available_val"], dtype=np.float32)  # (S,H,N,1)

    hd = hd / np.float32(scaler_std ** 2)
    hd = np.repeat(hd[:, None, :], horizon, axis=1)  # (S,H,N)
    fd = (fd / np.float32(scaler_std))[..., 0]  # (S,H,N)
    ex = ex[..., 0]  # (S,H,N)

    def squash(x):
        x = np.log1p(np.clip(x, 0.0, None))
        return x / (1.0 + x)

    return squash(hd), squash(fd), ex


def main() -> int:
    args = parse_args()
    ds_dir = args.dataset_dir
    report: dict = {
        "scope": "pems04_s1_validation_only_zero_training",
        "note": (
            "deployment-faithful emulator of --mode baseline "
            "--reranker-uniform under the frozen joint3 checkpoint"
        ),
    }

    index = np.load(ds_dir / f"{args.tag}_index_scaler.npz")
    idx_train = np.asarray(index["idx_train"], dtype=np.int64)
    idx_val = np.asarray(index["idx_val"], dtype=np.int64)
    scaler_mean = float(np.asarray(index["scaler_mean"]).item())
    scaler_std = float(np.asarray(index["scaler_std"]).item())

    from utils.datasets_utils import file_path_dict, load_raw_data

    raw, _ = load_raw_data(file_path_dict["PEMS04"])
    raw = raw[..., :1].astype(np.float32, copy=False)
    horizon = 12
    n_nodes = raw.shape[1]

    dump = np.load(args.backbone_dump, allow_pickle=True)
    p_val = np.asarray(dump["backbone_pred_val"], dtype=np.float32)  # (S,H,N,1)
    p_train = np.asarray(dump["backbone_pred_train"], dtype=np.float32)
    n_samples = p_val.shape[0]

    y_raw = raw[idx_val[:, None] + Y_OFFSETS[None, :], :, 0]  # (S,H,N)
    y_z = ((y_raw - scaler_mean) / scaler_std).astype(np.float32)
    valid_q = y_raw != 0  # the same validity convention the benchmark uses
    n_valid = int(valid_q.sum())

    cache = np.load(args.pca_cache, allow_pickle=True)
    temporal_mean = np.asarray(cache["retrieval_temporal_mean"], dtype=np.float32)
    temporal_basis = np.asarray(
        cache["retrieval_temporal_basis"], dtype=np.float32
    )
    e_val = np.asarray(cache["exact_retrieval_val"], dtype=np.float32)
    avail = np.asarray(cache["exact_available_val"], dtype=np.float32)
    q0, q1, q2 = quality_scalars(cache, scaler_std, horizon)
    t_canonical = decode_codes(
        cache["retrieval_code_val"], temporal_mean, temporal_basis
    )

    pipe = FrozenPipeline(args.joint3_exp, horizon, n_nodes)
    pipe.set_codec(temporal_mean, temporal_basis)
    report["gate_stats"] = {
        "g_mean": float(pipe.g.mean()),
        "g_min": float(pipe.g.min()),
        "g_max": float(pipe.g.max()),
        "exact_w_per_horizon": [round(float(v), 5) for v in pipe.w],
        "exact_available_fraction": float(avail.mean()),
    }

    def mae(prediction_z):
        err = np.abs(prediction_z[..., 0] - y_z)[valid_q]
        return float(err.mean()) * scaler_std

    # ---------------------------------------------------------------
    # JOB 1a -- acceptance test against recorded GPU benchmark numbers
    # ---------------------------------------------------------------
    if not args.skip_verify:
        print("=== emulator acceptance test ===")
        checks = []
        for name, expected in RECORDED.items():
            path = ds_dir / name
            if not path.exists():
                print(f"  skip (missing) {name}")
                continue
            with np.load(path, allow_pickle=True) as other:
                t_other = decode_codes(
                    other["retrieval_code_val"], temporal_mean, temporal_basis
                )
            got = mae(pipe.forward(p_val, t_other, e_val, avail, q0, q1, q2))
            checks.append(
                {
                    "cache": name,
                    "recorded": expected,
                    "emulated": round(got, 7),
                    "abs_delta": round(abs(got - expected), 7),
                }
            )
            print(
                f"  {name[40:]:55s} recorded={expected:.7f} "
                f"emulated={got:.7f} delta={got - expected:+.7f}"
            )
        report["acceptance_test"] = checks
        worst = max((c["abs_delta"] for c in checks), default=None)
        report["acceptance_test_max_abs_delta"] = worst
        print(f"  max |delta| = {worst}")
        if worst is None or worst > 5e-3:
            print(
                "EMULATOR DOES NOT REPRODUCE THE RECORDED BENCHMARK NUMBERS; "
                "refusing to build the oracle from it."
            )
            args.report.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return 2

    base_pred = pipe.forward(p_val, t_canonical, e_val, avail, q0, q1, q2)
    base_mae = (
        float((np.abs(base_pred[..., 0] - y_z) * valid_q).sum() / n_valid)
        * scaler_std
    )
    report["deployed_baseline_emulated"] = round(base_mae, 7)
    print(f"=== emulated canonical (uniform reranker) MAE = {base_mae:.7f}")

    # ---- optional: score arbitrary caches, for cross-checking a real run ----
    if args.eval_caches:
        extra = {}
        for path_str in args.eval_caches:
            path = Path(path_str)
            if not path.exists():
                extra[str(path)] = "missing"
                print(f"  eval-cache MISSING {path}")
                continue
            with np.load(path, allow_pickle=True) as other:
                t_other = decode_codes(
                    other["retrieval_code_val"], temporal_mean, temporal_basis
                )
            got = mae(pipe.forward(p_val, t_other, e_val, avail, q0, q1, q2))
            extra[path.name] = round(got, 7)
            print(f"  eval-cache {path.name}: emulated MAE = {got:.7f}")
        report["eval_caches"] = extra

    # ---------------------------------------------------------------
    # JOB 2 -- oracle at four selection granularities
    # ---------------------------------------------------------------
    with np.load(args.candidate_cache, allow_pickle=True) as cand_file:
        cand = np.asarray(cand_file["candidate_index_val"], dtype=np.int64)
    pos_lookup = build_pos_lookup(idx_train)
    n_cand = cand.shape[1]

    betas = sorted({float(b) for b in args.betas})
    build_beta = float(args.build_beta)
    if args.output_dir is not None:
        betas = sorted(set(betas) | {build_beta})

    totals = {
        b: {g: 0.0 for g in GRANULARITIES} for b in betas
    }
    rank_hist = {
        b: {g: np.zeros(n_cand, dtype=np.int64) for g in GRANULARITIES}
        for b in betas
    }
    t_oracle = (
        {g: t_canonical.copy() for g in args.build_granularities}
        if args.output_dir is not None
        else {}
    )
    reach_num = 0
    reach_den = 0

    for start in range(0, n_samples, args.chunk):
        end = min(n_samples, start + args.chunk)
        b_sz = end - start
        cand_blk = cand[start:end]

        y_f, nz = gather_candidate_future(
            raw, cand_blk, scaler_mean, scaler_std
        )  # (b,K,N,H) future in z space, (b,K,N,H) raw-nonzero
        y_hat = gather_candidate_parent(p_train, cand_blk, pos_lookup)[..., 0]
        r_i = np.transpose(y_f - y_hat, (0, 1, 3, 2))  # (b,K,H,N)
        ev = (
            np.transpose(nz, (0, 1, 3, 2))
            & (cand_blk >= 0)[:, :, None, :]
            & valid_q[start:end, None, :, :]
        )  # (b,K,H,N)

        p_blk = p_val[start:end]
        e_blk = e_val[start:end]
        a_blk = avail[start:end]
        y_blk = y_z[start:end]
        vq_blk = valid_q[start:end]
        q0b, q1b, q2b = q0[start:end], q1[start:end], q2[start:end]
        t_ref = t_canonical[start:end]

        vq_num = int(vq_blk.sum())
        reach_num += int((vq_blk & ev.any(axis=1)).sum())
        reach_den += vq_num

        b3 = np.arange(b_sz)[:, None, None]
        h3 = np.arange(horizon)[None, :, None]
        n3 = np.arange(n_nodes)[None, None, :]

        for b in betas:
            err = np.empty((b_sz, horizon, n_cand, n_nodes), dtype=np.float32)
            for k in range(n_cand):
                t_used = np.where(
                    ev[:, k][..., None],
                    p_blk + b * r_i[:, k, :, :, None],
                    t_ref,
                )
                pred = pipe.forward(p_blk, t_used, e_blk, a_blk, q0b, q1b, q2b)
                err[:, :, k, :] = np.abs(pred[..., 0] - y_blk) * vq_blk

            # --- per element: independent min over candidates -----------
            arg_el = np.argmin(err, axis=2)  # (b,H,N)
            totals[b]["element"] += float(
                np.take_along_axis(err, arg_el[:, :, None, :], axis=2).sum()
            )
            rank_hist[b]["element"] += np.bincount(
                arg_el.reshape(-1), minlength=n_cand
            )

            # --- per (query,horizon): min over K of the node sum --------
            node_sum = err.sum(axis=3)  # (b,H,K)
            arg_h = np.argmin(node_sum, axis=2)  # (b,H)
            totals[b]["query_horizon"] += float(
                np.take_along_axis(node_sum, arg_h[..., None], axis=2).sum()
            )
            rank_hist[b]["query_horizon"] += np.bincount(
                arg_h.reshape(-1), minlength=n_cand
            )

            # --- per (query,node): min over K of the horizon sum --------
            hor_sum = err.sum(axis=1)  # (b,K,N)
            arg_n = np.argmin(hor_sum, axis=1)  # (b,N)
            totals[b]["query_node"] += float(
                np.take_along_axis(hor_sum, arg_n[:, None, :], axis=1).sum()
            )
            rank_hist[b]["query_node"] += np.bincount(
                arg_n.reshape(-1), minlength=n_cand
            )

            # --- per query: one candidate for everything ----------------
            all_sum = err.sum(axis=(1, 3))  # (b,K)
            arg_q = np.argmin(all_sum, axis=1)  # (b,)
            totals[b]["query"] += float(
                all_sum[np.arange(b_sz), arg_q].sum()
            )
            rank_hist[b]["query"] += np.bincount(arg_q, minlength=n_cand)

            if args.output_dir is not None and b == build_beta:
                if "query_horizon" in t_oracle:
                    r_h = r_i[np.arange(b_sz)[:, None], arg_h, h3[:, :, 0], :]
                    ev_h = ev[
                        np.arange(b_sz)[:, None], arg_h, h3[:, :, 0], :,
                    ]
                    t_oracle["query_horizon"][start:end] = np.where(
                        ev_h[..., None], p_blk + b * r_h[..., None], t_ref
                    )
                if "element" in t_oracle:
                    r_e = r_i[b3, arg_el, h3, n3]
                    ev_e = ev[b3, arg_el, h3, n3]
                    t_oracle["element"][start:end] = np.where(
                        ev_e[..., None], p_blk + b * r_e[..., None], t_ref
                    )
        del y_f, nz, y_hat, r_i, ev, err

    results = {}
    best = None
    for b in betas:
        entry = {
            "emulated_canonical_mae": round(base_mae, 7),
            "granularities": {},
        }
        for g in GRANULARITIES:
            m = float(totals[b][g] / n_valid) * scaler_std
            hist = rank_hist[b][g]
            entry["granularities"][g] = {
                "mae": round(m, 7),
                "delta_vs_canonical": round(m - base_mae, 7),
                "chosen_rank_histogram": hist.tolist(),
                "rank0_fraction": round(float(hist[0] / hist.sum()), 5),
                "rank0_4_fraction": round(
                    float(hist[: min(5, n_cand)].sum() / hist.sum()), 5
                ),
            }
            print(
                f"  beta={b:<4g} {g:15s} MAE={m:.7f} "
                f"delta={m - base_mae:+.7f}"
            )
            if best is None or m < best[2]:
                best = (b, g, m)
        results[f"beta_{b:g}"] = entry

    report["oracle"] = results
    report["oracle_reachable_fraction"] = {
        "note": (
            "share of valid (q,h,n) elements where at least one shortlist "
            "candidate carries evidence (else the canonical cache value is "
            "kept)"
        ),
        "value": round(reach_num / max(reach_den, 1), 6),
    }
    report["oracle_best"] = {"beta": best[0], "granularity": best[1],
                             "mae": round(best[2], 7)}

    # ---------------------------------------------------------------
    # JOB 2b -- materialise the oracle caches
    # ---------------------------------------------------------------
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        built = {}
        for g in args.build_granularities:
            code = np.einsum(
                "shnc,nhr->srnc",
                (t_oracle[g] - temporal_mean[None]).astype(np.float32),
                temporal_basis,
                optimize=True,
            ).astype(np.float16)
            out_arrays = {k: cache[k] for k in cache.files}
            out_arrays["retrieval_code_val"] = code
            out_arrays["residual_construction"] = np.array(
                f"residual_oracle_{g}"
            )
            out_arrays["residual_construction_note"] = np.array(
                "channel0_val = P + beta*R_{argmin slot MAE}; uses query "
                "future, DIAGNOSTIC upper bound only; train/test codes copied "
                "verbatim from the canonical cache"
            )
            out_path = args.output_dir / (
                f"{args.tag}_analog_k8_exact_mean_oracle_{g}"
                f"_b{int(build_beta * 10)}_pca_r8_fp16.npz"
            )
            tmp = str(out_path) + ".tmp.npz"
            np.savez_compressed(tmp, **out_arrays)
            os.replace(tmp, out_path)
            built[g] = {
                "path": str(out_path),
                "predicted_mae": results[f"beta_{build_beta:g}"][
                    "granularities"
                ][g]["mae"],
            }
            print(f"Saved oracle cache ({g}): {out_path}")
        report["built_caches"] = {
            "beta": build_beta,
            "caches": built,
        }

    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "oracle"},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
