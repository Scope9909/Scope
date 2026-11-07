"""E2: fixed online-byte-budget SRM baselines on PEMS04 and PEMS08.

Accuracy uses one live full-memory and backbone Validation replay per dataset.
For a chosen node set, outputs are composed per node after checking the live
sparse replay against this identity. Serialized INT8 cold-start states measure
the actual bytes of every point. No Test arrays or new model training are used.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "e2"
sys.path.insert(0, str(ROOT))
from calibrate_dualscale_smooth_tod import fit_and_apply
from utils.online_residual_memory import packed_state

BENCH = ROOT / "paper_results/icde_retrieval_free_pilot_20261012/benchmark_online_end_to_end.py"
CONFIGS = {
    "PEMS04": {
        "exp": "EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_exppems04_joint3_lite_r256_strictalign_s1",
        "full": "cold_start_streaming_20261014/pems04_full_cold_state.npz",
        "sparse": "cold_start_streaming_20261014/pems04_sparse_cold_state.npz",
        "calibration": "paper_results/icde_generality/pems04_joint3_lite_r256_strictalign_s1_valcal/calibration.json",
    },
    "PEMS08": {
        "exp": "EXP_experiments/TLAST_improved/PEMS08/TLAST_improved_PEMS08_12to12_exppems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1",
        "full": "cold_start_streaming_20261014/pems08_full_cold_state.npz",
        "sparse": "cold_start_streaming_20261014/pems08_sparse_cold_state.npz",
        "calibration": "paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/calibration.json",
    },
}
FOLDS = ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0))
BUDGETS = (0.05, 0.10, 0.20, 0.40)
RANDOM_SEEDS = tuple(range(20261015, 20261025))


def save(path: Path, obj: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def benchmark(dataset: str, cfg: dict, mode: str, state: Path | None,
              tag: str, batch_size: int, warmup: int, export_prediction: bool) -> Path | None:
    folder = OUT / dataset.lower()
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / f"{tag}.json"
    prediction = folder / f"{tag}_predictions.npz" if export_prediction else None
    if result.exists() and (prediction is None or prediction.exists()):
        print(f"E2 {dataset} {tag} reuse", flush=True)
        return prediction
    if result.exists() or (prediction is not None and prediction.exists()):
        raise FileExistsError(f"Incomplete existing benchmark: {tag}")
    command = [sys.executable, "-B", str(BENCH), "--exp-dir", str(ROOT / cfg["exp"]),
               "--mode", mode, "--batch-size", str(batch_size),
               "--warmup-passes", str(warmup), "--repeats", "1", "--output", str(result)]
    if state is not None:
        command.extend(("--state", str(state)))
    if prediction is not None:
        command.extend(("--predictions-output", str(prediction)))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "_runtime_deps"), str(ROOT),
                                          env.get("PYTHONPATH", "")))
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    print(f"E2 {dataset} {tag} begin", flush=True)
    with (folder / f"{tag}.stdout.log").open("w", encoding="utf-8") as stdout, \
         (folder / f"{tag}.stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                   stderr=stderr, creationflags=flags, check=False)
    if completed.returncode or not result.exists() or (prediction and not prediction.exists()):
        raise RuntimeError(f"E2 {dataset} {tag} failed; inspect its stderr log")
    record = json.loads(result.read_text(encoding="utf-8"))
    if record.get("test_evaluated") or record.get("query_cache_loaded"):
        raise ValueError(f"E2 {dataset} {tag}: benchmark protocol violation")
    print(f"E2 {dataset} {tag} done", flush=True)
    return prediction


def read_prediction(path: Path):
    with np.load(path, allow_pickle=False) as a:
        return (np.asarray(a["val_prediction"], dtype=np.float32),
                np.asarray(a["val_truth"], dtype=np.float32),
                np.asarray(a["val_endpoints"], dtype=np.int64))


def state_subset(full: dict, ids: np.ndarray) -> dict:
    ids = np.sort(np.asarray(ids, dtype=np.int32))
    out = dict(full)
    out["node_ids"] = ids
    out["train_raw"] = np.ascontiguousarray(full["train_raw"][:, ids])
    out["quant_scale"] = np.ascontiguousarray(full["quant_scale"][:, ids, :])
    for key in ("scaler_mean", "scaler_std"):
        if np.asarray(full[key]).size > 1:
            out[key] = np.asarray(full[key])[ids]
    return out


def choose(order: np.ndarray, proxy: np.ndarray, full: dict, cap: int,
           empty_bytes: int, positive: np.ndarray | None = None):
    selected = []
    estimate = empty_bytes
    for node in order:
        node = int(node)
        if positive is not None and not positive[node]:
            continue
        if estimate + proxy[node] <= cap:
            selected.append(node)
            estimate += int(proxy[node])
    while selected:
        payload = packed_state(state_subset(full, np.asarray(selected)))
        if len(payload) <= cap:
            return np.sort(np.asarray(selected, dtype=np.int32)), payload
        selected.pop()
    return np.empty(0, dtype=np.int32), b""


def loss_table(pred: np.ndarray, truth: np.ndarray, start: int, end: int):
    target = truth[start:end]
    valid = np.rint(target) > 0
    return (np.where(valid, np.abs(pred[start:end] - target), 0).sum(axis=0, dtype=np.float64),
            valid.sum(axis=0, dtype=np.int64))


def score(base_sum, full_sum, count, ids):
    selected = np.zeros(base_sum.shape[1], dtype=bool)
    selected[ids] = True
    combined = np.where(selected[None, :], full_sum, base_sum)
    return float(np.mean(combined.sum(axis=1) / count.sum(axis=1)))


def run_dataset(dataset: str, cfg: dict) -> dict:
    folder = OUT / dataset.lower()
    state_full_path = ROOT / cfg["full"]
    state_sparse_path = ROOT / cfg["sparse"]
    for path in (state_full_path, state_sparse_path, ROOT / cfg["exp"] / "best.pth"):
        if not path.exists():
            raise FileNotFoundError(path)
    paths = {}
    for mode in ("backbone", "full", "sparse"):
        state = None if mode == "backbone" else (state_full_path if mode == "full" else state_sparse_path)
        paths[mode] = benchmark(dataset, cfg, mode, state, f"{mode}_b32",
                                32, 0, True)
    base, truth, endpoints = read_prediction(paths["backbone"])
    full_pred, full_truth, full_endpoints = read_prediction(paths["full"])
    sparse_pred, sparse_truth, sparse_endpoints = read_prediction(paths["sparse"])
    if not (np.array_equal(truth, full_truth) and np.array_equal(truth, sparse_truth)
            and np.array_equal(endpoints, full_endpoints)
            and np.array_equal(endpoints, sparse_endpoints)):
        raise ValueError(f"{dataset}: online replay targets/endpoints differ")
    with np.load(state_full_path, allow_pickle=False) as a:
        full_state = {key: a[key] for key in a.files}
    with np.load(state_sparse_path, allow_pickle=False) as a:
        existing_ids = np.asarray(a["node_ids"], dtype=np.int32)
    node_count = truth.shape[2]
    if not np.array_equal(full_state["node_ids"], np.arange(node_count)):
        raise ValueError(f"{dataset}: full state does not cover ordered nodes")
    reconstructed = base.copy()
    reconstructed[:, :, existing_ids] = full_pred[:, :, existing_ids]
    delta = np.abs(reconstructed - sparse_pred)
    parity = {"max_abs": float(delta.max()), "p999_abs": float(np.quantile(delta, 0.999)),
              "mean_abs": float(delta.mean())}
    if parity["p999_abs"] > 1e-3 or parity["mean_abs"] > 1e-4:
        raise ValueError(f"{dataset}: per-node live composition failed: {parity}")
    fit_end = int(len(truth) * 0.55)
    base_fit, count = loss_table(base, truth, 0, fit_end)
    full_fit, count2 = loss_table(full_pred, truth, 0, fit_end)
    if not np.array_equal(count, count2):
        raise ValueError("Fit mask differs")
    gain = (base_fit - full_fit).sum(axis=0) / count.sum(axis=0)
    raw = np.asarray(full_state["train_raw"])
    scale = np.asarray(full_state["quant_scale"])
    proxy = np.array([len(zlib.compress(np.ascontiguousarray(raw[:, i]).tobytes(), 6)) +
                      len(zlib.compress(np.ascontiguousarray(scale[:, i, :]).tobytes(), 6))
                      for i in range(node_count)], dtype=np.int64)
    empty_bytes = len(packed_state(state_subset(full_state, np.empty(0, dtype=np.int32))))
    top_order = np.argsort(-gain, kind="stable")
    ratio_order = np.lexsort((np.arange(node_count), -gain / proxy))
    calibration = json.loads((ROOT / cfg["calibration"]).read_text(encoding="utf-8"))
    params = {key: calibration[key] for key in ("num_bins", "smooth_shrinkage",
              "num_time_bins", "time_shrinkage", "slices_per_day")}
    fold_loss = []
    for fit_frac, eval_frac in FOLDS:
        fit, end = int(len(truth) * fit_frac), int(len(truth) * eval_frac)
        _, base_cal, _ = fit_and_apply(base[:fit], truth[:fit], endpoints[:fit],
                                       base[fit:end], endpoints[fit:end], **params)
        _, full_cal, _ = fit_and_apply(full_pred[:fit], truth[:fit], endpoints[:fit],
                                       full_pred[fit:end], endpoints[fit:end], **params)
        raw_base, counts = loss_table(base, truth, fit, end)
        raw_full, counts2 = loss_table(full_pred, truth, fit, end)
        target = truth[fit:end]
        mask = np.rint(target) > 0
        cal_base = np.where(mask, np.abs(base_cal - target), 0).sum(axis=0, dtype=np.float64)
        cal_full = np.where(mask, np.abs(full_cal - target), 0).sum(axis=0, dtype=np.float64)
        if not np.array_equal(counts, counts2):
            raise ValueError("Fold mask differs")
        fold_loss.append((fit, end, raw_base, raw_full, cal_base, cal_full, counts))
        print(f"E2 {dataset} calibration fold {len(fold_loss)}/3", flush=True)
    full_bytes = state_full_path.stat().st_size
    result = {"dataset": dataset, "seed": 1, "test_evaluated": False,
              "full_cold_state_bytes": full_bytes, "empty_state_bytes": empty_bytes,
              "full_state": str(state_full_path), "live_composition_parity": parity,
              "selection_prefix_samples": fit_end,
              "budget_fractions": BUDGETS, "random_seeds": RANDOM_SEEDS,
              "fit_gain_by_node": gain.tolist(), "proxy_cost_by_node": proxy.tolist(),
              "rows": []}

    def add_row(method, fraction, ids, payload=None, random_seed=None):
        if payload is None:
            payload = packed_state(state_subset(full_state, ids))
        folds = [{"eval_start": fit, "eval_end": end,
                  "raw_mae": score(rb, rf, c, ids),
                  "calibrated_mae": score(cb, cf, c, ids)}
                 for fit, end, rb, rf, cb, cf, c in fold_loss]
        row = {"method": method, "fraction": fraction, "random_seed": random_seed,
               "selected_nodes": len(ids), "node_ids": ids.tolist(),
               "state_bytes": len(payload), "folds": folds,
               "mean_raw_mae": float(np.mean([f["raw_mae"] for f in folds])),
               "mean_calibrated_mae": float(np.mean([f["calibrated_mae"] for f in folds]))}
        result["rows"].append(row)
        return row

    add_row("backbone", None, np.empty(0, dtype=np.int32), b"")
    add_row("full", 1.0, np.arange(node_count, dtype=np.int32),
            packed_state(full_state))
    add_row("current_frozen", None, existing_ids)
    for fraction in BUDGETS:
        cap = int(full_bytes * fraction)
        if cap <= empty_bytes:
            result["rows"].append({"fraction": fraction, "status": "infeasible metadata floor"})
            continue
        methods = [("top_gain", top_order, None),
                   ("ordinary_gain_per_byte", ratio_order, gain > 0)]
        for method, order, positive in methods:
            ids, payload = choose(order, proxy, full_state, cap, empty_bytes, positive)
            if not len(ids):
                raise ValueError(f"{dataset}/{fraction}/{method}: no nodes fit")
            row = add_row(method, fraction, ids, payload)
            row["byte_cap"] = cap
            row["unused_bytes"] = cap - len(payload)
            if fraction in (0.10, 0.40):
                state_path = folder / f"{method}_{int(100*fraction)}pct_cold_state.npz"
                if state_path.exists():
                    raise FileExistsError(state_path)
                state_path.write_bytes(payload)
                row["state_path"] = str(state_path)
        for random_seed in RANDOM_SEEDS:
            order = np.random.default_rng(random_seed).permutation(node_count)
            ids, payload = choose(order, proxy, full_state, cap, empty_bytes)
            row = add_row("random", fraction, ids, payload, random_seed)
            row["byte_cap"] = cap
            row["unused_bytes"] = cap - len(payload)
        print(f"E2 {dataset} budget {fraction:.0%} accuracy done", flush=True)
    save(folder / "accuracy_bytes.json", result)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": "E2_frozen_seed1_online_state_same_byte_budget_v1",
               "split": "Validation only", "test_evaluated": False,
               "budget_fractions": BUDGETS,
               "selection": "first 55% Validation; pre-fixed top-gain, ordinary positive-gain/byte and ten random seeds",
               "calibration": "same expanding-time folds; model-specific per-node smooth and target-time fit",
               "accuracy": "full and backbone live replay; per-node composition checked against independent live sparse replay",
               "timing": "batch=1 full Validation at 10% and 40% for top-gain and gain/byte; baseline/full endpoints",
               "datasets": {}}
    for dataset, cfg in CONFIGS.items():
        row = run_dataset(dataset, cfg)
        folder = OUT / dataset.lower()
        for mode in ("backbone", "full"):
            state = None if mode == "backbone" else ROOT / cfg["full"]
            benchmark(dataset, cfg, mode, state, f"{mode}_b1", 1, 1, False)
        for fraction in (0.10, 0.40):
            for method in ("top_gain", "ordinary_gain_per_byte"):
                state = folder / f"{method}_{int(100*fraction)}pct_cold_state.npz"
                benchmark(dataset, cfg, "sparse", state,
                          f"{method}_{int(100*fraction)}pct_b1", 1, 1, False)
        summary["datasets"][dataset] = {"accuracy_bytes": str(folder / "accuracy_bytes.json"),
                                        "timing_results": sorted(str(p) for p in folder.glob("*_b1.json"))}
        print(f"E2 {dataset} complete", flush=True)
    save(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
