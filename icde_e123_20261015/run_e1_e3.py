"""Matched calibration (E1) and paired moving-block bootstrap (E3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from calibrate_dualscale_quantile_residual import metrics
from calibrate_dualscale_smooth_tod import fit_and_apply

FOLDS = ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0))
E1 = {
    "PEMS04": ("paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_s1_valcal",
               "datasets/PEMS04/PEMS04_12to12_parent_predictions_joint3.npz"),
    "PEMS07": ("paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal",
               "datasets/PEMS07/PEMS07_12to12_parent_predictions_joint3.npz"),
}
E3 = {
    "PEMS03": ("datasets/PEMS03/PEMS03_12to12_parent_predictions_joint3.npz",
               "paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz"),
    "PEMS04": ("datasets/PEMS04/PEMS04_12to12_parent_predictions_joint3.npz",
               "paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_s1_valcal/selected_predictions.npz"),
    "PEMS07": ("datasets/PEMS07/PEMS07_12to12_parent_predictions_joint3.npz",
               "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz"),
    "PEMS08": ("datasets/PEMS08/PEMS08_12to12_parent_predictions_joint3.npz",
               "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz"),
    "PEMSBAY": ("datasets/PEMSBAY/PEMSBAY_12to12_parent_predictions_final_joint_712.npz",
                "paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz"),
}


def save(path: Path, obj: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def split(path: Path, name: str):
    with np.load(path, allow_pickle=False) as a:
        return (np.asarray(a[f"{name}_prediction"], dtype=np.float32),
                np.asarray(a[f"{name}_truth"], dtype=np.float32),
                np.asarray(a[f"{name}_endpoints"], dtype=np.int64))


def teacher(path: Path, name: str, count: int):
    with np.load(path, allow_pickle=False) as a:
        z = np.asarray(a[f"teacher_pred_{name}"], dtype=np.float32)
        mean = np.asarray(a["scaler_mean"], dtype=np.float32)
        std = np.asarray(a["scaler_std"], dtype=np.float32)
    if len(z) < count:
        raise ValueError(f"Teacher cache shorter than official {name}: {path}")
    padding = len(z) - count
    z = z[:count]
    if z.ndim == 4 and z.shape[-1] == 1:
        z = z[..., 0]
    if mean.size == 1:
        p = z * std.item() + mean.item()
    else:
        p = z * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    return p.astype(np.float32), padding


def mae_affine(p: np.ndarray, y: np.ndarray):
    valid = np.rint(y) > 0
    x, target = p[valid], y[valid]
    if not len(x):
        raise ValueError("No valid calibration labels")
    best = (float("inf"), 1.0, 0.0)
    for grid in (np.linspace(0.8, 1.2, 9), None):
        if grid is None:
            grid = np.linspace(max(0.8, best[1] - 0.05), min(1.2, best[1] + 0.05), 9)
        for slope in grid:
            residual = target - np.float32(slope) * x
            bias = float(np.median(residual))
            error = float(np.mean(np.abs(residual - np.float32(bias))))
            if error < best[0]:
                best = error, float(slope), bias
    return best[1:]


def run_e1() -> None:
    summary = {"protocol": "E1_equal_validation_folds_seed1_v1", "test_opened": False,
               "affine": "global a*yhat+b; a in [0.8,1.2], fixed 9+9 slope grid, b median residual",
               "datasets": {}}
    for dataset, (cal_rel, parent_rel) in E1.items():
        cal = json.loads((ROOT / cal_rel / "calibration.json").read_text(encoding="utf-8"))
        raw_path = Path(cal["source_predictions"])
        if not raw_path.exists():
            raw_path = ROOT / raw_path
        student_pred, truth, endpoints = split(raw_path, "val")
        parent_pred, padding = teacher(ROOT / parent_rel, "val", len(truth))
        if parent_pred.shape != student_pred.shape or student_pred.shape != truth.shape:
            raise ValueError(f"{dataset}: Validation shape mismatch")
        if not np.all(np.diff(endpoints) > 0):
            raise ValueError(f"{dataset}: non-monotonic endpoints")
        params = {key: cal[key] for key in ("num_bins", "smooth_shrinkage",
                  "num_time_bins", "time_shrinkage", "slices_per_day")}
        models = {}
        for name, pred in (("Teacher", parent_pred), ("Student", student_pred)):
            folds = []
            for f, end in FOLDS:
                fit_end, eval_end = int(len(truth) * f), int(len(truth) * end)
                query, target = pred[fit_end:eval_end], truth[fit_end:eval_end]
                _, calibrated, tables = fit_and_apply(
                    pred[:fit_end], truth[:fit_end], endpoints[:fit_end],
                    query, endpoints[fit_end:eval_end], **params)
                slope, bias = mae_affine(pred[:fit_end], truth[:fit_end])
                folds.append({"fit_end": fit_end, "eval_end": eval_end,
                              "none": metrics(query, target),
                              "smooth_target_time": metrics(calibrated, target),
                              "mae_affine": metrics(slope * query + bias, target),
                              "slope": slope, "bias": bias,
                              "smooth_table_values": int(sum(t.size for t in tables))})
                print(f"E1 {dataset} {name} fold {len(folds)}/3", flush=True)
            models[name] = {"folds": folds,
                            "mean_fold_mae": {k: float(np.mean([r[k]["mae"] for r in folds]))
                                              for k in ("none", "smooth_target_time", "mae_affine")}}
        row = {"student_source": str(raw_path.resolve()),
               "teacher_source": str((ROOT / parent_rel).resolve()),
               "teacher_padding_discarded": padding, "models": models,
               "student_minus_teacher_mae": {
                   k: models["Student"]["mean_fold_mae"][k] - models["Teacher"]["mean_fold_mae"][k]
                   for k in ("none", "smooth_target_time", "mae_affine")}}
        summary["datasets"][dataset] = row
        save(OUT / f"e1_{dataset.lower()}.json", row)
        del student_pred, parent_pred, truth
    save(OUT / "e1_summary.json", summary)


def sums(pred: np.ndarray, truth: np.ndarray):
    mask = np.rint(truth) > 0
    return (np.where(mask, np.abs(pred - truth), 0).sum(axis=2, dtype=np.float64),
            mask.sum(axis=2, dtype=np.int64))


def starts(endpoints: np.ndarray, length: int):
    boundaries = np.r_[0, np.flatnonzero(np.diff(endpoints) != 1) + 1, len(endpoints)]
    spans = [np.arange(a, b - length + 1, dtype=np.int64)
             for a, b in zip(boundaries[:-1], boundaries[1:]) if b - a >= length]
    return np.concatenate(spans) if spans else np.empty(0, dtype=np.int64)


def run_e3() -> None:
    summary = {"protocol": "E3_paired_moving_block_bootstrap_v1", "repeats": 5000,
               "block_lengths": [12, 72, 288], "seed": 20261015,
               "split": "official Test; existing frozen predictions",
               "metric": "round(truth)>0 masked per-horizon MAE, then horizon mean",
               "difference": "Teacher minus Student+ValCal", "datasets": {}}
    for offset, (dataset, (parent_rel, final_rel)) in enumerate(E3.items()):
        final, truth, endpoints = split(ROOT / final_rel, "test")
        parent, padding = teacher(ROOT / parent_rel, "test", len(truth))
        if parent.shape != final.shape or final.shape != truth.shape:
            raise ValueError(f"{dataset}: Test shape mismatch")
        parent_sum, counts = sums(parent, truth)
        final_sum, counts2 = sums(final, truth)
        if not np.array_equal(counts, counts2):
            raise ValueError("Paired masks differ")
        observed = float(np.mean(parent_sum.sum(axis=0) / counts.sum(axis=0) -
                                 final_sum.sum(axis=0) / counts.sum(axis=0)))
        blocks = {}
        n = len(truth)
        for length in (12, 72, 288):
            possible = starts(endpoints, length)
            if not len(possible):
                blocks[str(length)] = {"available": False, "reason": "no continuous segment"}
                continue
            rng = np.random.default_rng(20261015 + offset * 1000 + length)
            draws = np.empty(5000)
            offsets = np.arange(length)
            number = (n + length - 1) // length
            for begin in range(0, 5000, 20):
                batch = min(20, 5000 - begin)
                chosen = rng.choice(possible, size=(batch, number), replace=True)
                ix = (chosen[:, :, None] + offsets).reshape(batch, -1)[:, :n]
                weight = counts[ix].sum(axis=1)
                draws[begin:begin + batch] = np.mean(
                    parent_sum[ix].sum(axis=1) / weight - final_sum[ix].sum(axis=1) / weight,
                    axis=1)
            blocks[str(length)] = {"available": True, "valid_block_starts": len(possible),
                                   "independent_blocks_approx": n // length,
                                   "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                                   "probability_improvement_le_zero": float(np.mean(draws <= 0))}
            print(f"E3 {dataset} block {length}", flush=True)
        row = {"samples": n, "teacher_padding_discarded": padding,
               "teacher_mae": float(np.mean(parent_sum.sum(axis=0) / counts.sum(axis=0))),
               "student_valcal_mae": float(np.mean(final_sum.sum(axis=0) / counts.sum(axis=0))),
               "observed_teacher_minus_student": observed, "blocks": blocks,
               "teacher_source": str((ROOT / parent_rel).resolve()),
               "student_source": str((ROOT / final_rel).resolve())}
        summary["datasets"][dataset] = row
        save(OUT / f"e3_{dataset.lower()}.json", row)
        del final, parent, truth, parent_sum, final_sum
    save(OUT / "e3_summary.json", summary)


if __name__ == "__main__":
    run_e1()
    run_e3()
