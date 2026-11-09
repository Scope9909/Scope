"""Validation-only SRM follow-up: PEMS08 attribution and isolated timing repeats."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_runtime_deps"))
import numpy as np

BENCHMARK = ROOT / "paper_results" / "icde_retrieval_free_pilot" / "benchmark_online_end_to_end.py"
PREVIOUS = ROOT / "srm_validation_diagnostics" / "results.json"
CONFIGS = {
    "PEMS08": ("TLAST_improved_PEMS08_12to12_exppems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1",
               "cold_start_streaming/pems08_{mode}_cold_state.npz"),
    "PEMS03": ("TLAST_improved_PEMS03_12to12_exp10101500_s1",
               "pems03_int8_uniform/{mode}_cold_state.npz"),
    "PEMS04": ("TLAST_improved_PEMS04_12to12_exppems04_joint3_lite_r256_strictalign_s1",
               "cold_start_streaming/pems04_{mode}_cold_state.npz"),
    "PEMS07": ("TLAST_improved_PEMS07_12to12_exppems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1",
               "cold_start_streaming/pems07_{mode}_cold_state.npz"),
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def inputs(dataset: str, mode: str) -> tuple[Path, Path]:
    experiment, pattern = CONFIGS[dataset]
    exp = ROOT / "EXP_experiments" / "TLAST_improved" / dataset / experiment
    state = ROOT / pattern.format(mode=mode)
    for path in (exp / "best.pth", exp / "config.pkl", state):
        if not path.is_file():
            raise FileNotFoundError(path)
    return exp, state


def preflight() -> dict:
    if not BENCHMARK.is_file():
        raise FileNotFoundError(BENCHMARK)
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    if previous["status"] != "complete" or set(previous["results"]) != set(CONFIGS):
        raise ValueError("Prior four-dataset Validation diagnostic is not complete")
    checked = {}
    for dataset in CONFIGS:
        for mode in ("sparse", "full"):
            exp, state = inputs(dataset, mode)
            checked[f"{dataset}/{mode}"] = {
                "checkpoint": str(exp / "best.pth"), "state": str(state),
                "state_bytes": state.stat().st_size,
            }
    return checked


def per_node_and_day_attribution(sparse_path: Path, full_path: Path) -> dict:
    with np.load(sparse_path, allow_pickle=False) as archive:
        sparse = np.asarray(archive["val_prediction"], dtype=np.float64)
        truth = np.asarray(archive["val_truth"], dtype=np.float64)
        endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
    with np.load(full_path, allow_pickle=False) as archive:
        full = np.asarray(archive["val_prediction"], dtype=np.float64)
        if not np.array_equal(truth, archive["val_truth"]):
            raise ValueError("PEMS08 sparse/full Validation truth differs")
        if not np.array_equal(endpoints, archive["val_endpoints"]):
            raise ValueError("PEMS08 sparse/full Validation endpoints differ")
    if sparse.shape != full.shape or sparse.shape != truth.shape:
        raise ValueError("PEMS08 prediction shapes differ")
    n, horizons, nodes = truth.shape
    if horizons != 12 or nodes != 170:
        raise ValueError(f"Unexpected PEMS08 shape {truth.shape}")
    _, sparse_state = inputs("PEMS08", "sparse")
    with np.load(sparse_state, allow_pickle=False) as archive:
        selected_ids = np.asarray(archive["node_ids"], dtype=np.int64)
    selected = np.zeros(nodes, dtype=bool)
    selected[selected_ids] = True
    valid = np.round(truth) > 0
    sparse_error = np.abs(sparse - truth)
    full_error = np.abs(full - truth)
    delta = np.where(valid, sparse_error - full_error, 0.0)
    if np.max(np.abs(sparse[:, :, selected] - full[:, :, selected])) > 1e-5:
        raise ValueError("Selected-node predictions differ under frozen sparse/full states")
    tail_start = int(n * 0.55)
    tail_delta = delta[tail_start:]
    tail_valid = valid[tail_start:]
    per_node_sum = tail_delta.sum(axis=(0, 1))
    per_node_count = tail_valid.sum(axis=(0, 1))
    per_node = []
    positive_sum = float(np.maximum(per_node_sum, 0).sum())
    for node in range(nodes):
        count = int(per_node_count[node])
        per_node.append({
            "node": node, "selected": bool(selected[node]),
            "valid_count": count,
            "signed_absolute_error_change_sum": float(per_node_sum[node]),
            "sparse_minus_full_mae": float(per_node_sum[node] / count) if count else None,
            "share_of_positive_added_error": (float(max(per_node_sum[node], 0) / positive_sum)
                                               if positive_sum else 0.0),
        })
    days = []
    day_ids = endpoints // 288  # Dataset-relative 5-minute day block, not a calendar date.
    for day in np.unique(day_ids[tail_start:]):
        mask = (day_ids == day) & (np.arange(n) >= tail_start)
        count = int(valid[mask].sum())
        days.append({
            "dataset_relative_day_block": int(day), "samples": int(mask.sum()),
            "valid_count": count,
            "sparse_minus_full_mae": float(delta[mask].sum() / count) if count else None,
        })
    folds = []
    boundaries = [int(n * f) for f in (0.55, 0.70, 0.85, 1.0)]
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        count = int(valid[start:end].sum())
        folds.append({"fold": index, "start": start, "end": end,
                      "sparse_minus_full_mae": float(delta[start:end].sum() / count)})
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    expected = previous["results"]["PEMS08"]["modes"]
    for row in folds:
        key = f"fold{row['fold']}/all/all/all"
        reference = expected["sparse"]["stats"][key]["mae"] - expected["full"]["stats"][key]["mae"]
        if abs(row["sparse_minus_full_mae"] - reference) > 1e-4:
            raise ValueError(f"Fold {row['fold']} does not reproduce prior diagnostic")
    return {
        "protocol": "pems08_frozen_seed1_validation_pairwise_attribution_v1",
        "validation_samples": n, "tail_start": tail_start,
        "selected_node_count": int(selected.sum()),
        "tail_sparse_minus_full_mae": float(tail_delta.sum() / tail_valid.sum()),
        "folds": folds,
        "day_blocks": days,
        "per_node": per_node,
        "top_positive_unselected_nodes": sorted(
            (row for row in per_node if not row["selected"]),
            key=lambda row: row["signed_absolute_error_change_sum"], reverse=True)[:15],
        "note": "No node reselection. Day IDs are dataset-relative 288-slot blocks, not calendar dates.",
    }


def render(status: dict) -> str:
    lines = [
        "# SRM 后续 Validation：PEMS08 误差归因与重复在线测量", "",
        f"状态：**{status['status']}**；已完成测量 {len(status['jobs'])}/40；更新时间 {status['updated_at']}。", "",
        "本轮延续已冻结方案，不训练、不重新选点、不拟合新校准器，**不评估 Test**。所有模式使用 seed=1 已有 checkpoint 和 Train-only 冷启动状态。", "",
        "## 目的与设计", "",
        "1. 固定 PEMS08 sparse/full，逐节点和按数据集相对日期块计算后 45% Validation 的配对绝对误差差值，定位已观察到的 MAE 代价；只做归因，不用同一切片回头改选点。", "",
        "2. PEMS03/04/07/08 各对 sparse/full 做 5 次**独立进程**重放。每次 1 遍完整 Validation 预热、1 遍计时，batch=32，分别记录检索累计时间、端到端全程时间、采样进程 RSS。报告中位数及最小–最大范围，不把 RSS 差解释为纯记忆占用。", "",
        "3. 这测的是顺序 batch=32 重放；不能替代既有 batch=1 单请求 p50/p95，也不代表并发吞吐。PEMS07 曾参与规则设计；本轮只有 seed=1，不能当独立数据集或独立训练种子泛化。", "",
        "## PEMS08 固定方案误差归因", "",
    ]
    attr = status.get("attribution")
    if attr:
        lines += [f"后 45% Validation 的 sparse−full 配对 micro-MAE：**{attr['tail_sparse_minus_full_mae']:+.6f}**。", "",
                  "| 后续折 | Sparse−Full raw MAE |", "|---|---:|"]
        for row in attr["folds"]:
            lines.append(f"| {row['fold']} | {row['sparse_minus_full_mae']:+.6f} |")
        lines += ["", "未选中节点中，按累计增加的绝对误差排序（分母是所有正向增加的节点误差和，负向改善可抵消净 MAE）：", "",
                  "| 节点 | 节点 MAE 差 | 正向新增误差占比 |", "|---:|---:|---:|"]
        for row in attr["top_positive_unselected_nodes"][:10]:
            lines.append(f"| {row['node']} | {row['sparse_minus_full_mae']:+.6f} | {100*row['share_of_positive_added_error']:.2f}% |")
        lines += ["", "| 数据集相对日块 | 样本数 | Sparse−Full raw MAE |", "|---:|---:|---:|"]
        for row in attr["day_blocks"]:
            lines.append(f"| {row['dataset_relative_day_block']} | {row['samples']} | {row['sparse_minus_full_mae']:+.6f} |")
        lines += ["", "完整 170 节点统计见 `pems08_attribution.json`。日期块不是实际公历日期，不能用它推断节假日原因。", ""]
    else:
        lines += ["等待 PEMS08 sparse/full 首次配对预测完成。", ""]
    lines += ["## 五次独立进程重放", "",
              "| 数据集 | 模式 | 已完成次数 | 检索秒数：中位数 [最小, 最大] | 全程秒数：中位数 [最小, 最大] | 采样 RSS 峰值：中位数 MiB |",
              "|---|---|---:|---:|---:|---:|"]
    for dataset in CONFIGS:
        for mode in ("sparse", "full"):
            rows = [row for row in status["jobs"] if row["dataset"] == dataset and row["mode"] == mode]
            if not rows:
                lines.append(f"| {dataset} | {mode} | 0/5 | — | — | — |")
                continue
            retrieval = np.asarray([row["retrieval_seconds"] for row in rows], dtype=float)
            wall = np.asarray([row["wall_seconds"] for row in rows], dtype=float)
            rss = np.asarray([row["rss_peak_bytes"] for row in rows], dtype=float) / 1024**2
            lines.append(f"| {dataset} | {mode} | {len(rows)}/5 | "
                         f"{np.median(retrieval):.2f} [{retrieval.min():.2f}, {retrieval.max():.2f}] | "
                         f"{np.median(wall):.2f} [{wall.min():.2f}, {wall.max():.2f}] | "
                         f"{np.median(rss):.1f} |")
    if status.get("error"):
        lines += ["", "## 错误", "", "```text", status["error"], "```"]
    lines += ["", "## 写作边界", "",
              "即使多次计时稳定，也只能说明当前硬件、顺序 Validation 重放下的成本范围。PEMS08 的节点/日块切片用于解释反例，不是用于同折调参与再验证。要证明新选点算法优于强基线，还需要预先冻结规则、相同完整部署字节预算和未参与设计的数据/时间段。", ""]
    return "\n".join(lines)


def save(status: dict) -> None:
    status["updated_at"] = now()
    write_json(OUT / "status.json", status)
    temp = OUT / "SRM_FOLLOWUP_VALIDATION_20261015.md.tmp"
    temp.write_text(render(status), encoding="utf-8")
    temp.replace(OUT / "SRM_FOLLOWUP_VALIDATION_20261015.md")


def run_job(dataset: str, mode: str, repeat: int) -> dict:
    exp, state = inputs(dataset, mode)
    prefix = f"{dataset.lower()}_{mode}_r{repeat}"
    output = OUT / f"{prefix}.json"
    stdout = OUT / f"{prefix}.stdout.log"
    stderr = OUT / f"{prefix}.stderr.log"
    prediction_path = OUT / f"{prefix}_predictions.npz" if dataset == "PEMS08" and repeat == 1 else None
    if output.exists() or (prediction_path and prediction_path.exists()):
        raise FileExistsError(f"Refusing to overwrite prior job: {prefix}")
    command = [sys.executable, "-B", str(BENCHMARK),
               "--exp-dir", str(exp), "--state", str(state), "--mode", mode,
               "--batch-size", "32", "--warmup-passes", "1", "--repeats", "1",
               "--output", str(output)]
    if prediction_path:
        command += ["--predictions-output", str(prediction_path)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "_runtime_deps"), str(ROOT),
                                          env.get("PYTHONPATH", "")))
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, cwd=ROOT, env=env,
                                   stdout=out, stderr=err, creationflags=flags,
                                   check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"{prefix} failed with exit {completed.returncode}; see {stderr}")
    result = json.loads(output.read_text(encoding="utf-8"))
    if result["test_evaluated"] or result["query_cache_loaded"]:
        raise ValueError(f"{prefix}: protocol violation")
    if result["dataset"] != dataset or result["mode"] != mode or result["samples_per_pass"] < 1000:
        raise ValueError(f"{prefix}: incomplete Validation pass")
    return {
        "dataset": dataset, "mode": mode, "repeat": repeat,
        "retrieval_seconds": result["retrieval_seconds"][0],
        "wall_seconds": result["pass_wall_seconds"][0],
        "rss_peak_bytes": result["cpu_rss_peak_sampled_bytes"],
        "state_bytes": result["state_bytes"],
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    checked = preflight()
    if args.preflight:
        print(json.dumps(checked, ensure_ascii=False, indent=2))
        return
    if args.smoke:
        exp, state = inputs("PEMS08", "sparse")
        output = OUT / "smoke_pems08_sparse_64.json"
        if output.exists():
            raise FileExistsError(output)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "_runtime_deps"), str(ROOT),
                                              env.get("PYTHONPATH", "")))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.run([sys.executable, "-B", str(BENCHMARK),
                        "--exp-dir", str(exp), "--state", str(state),
                        "--mode", "sparse", "--batch-size", "32",
                        "--warmup-passes", "0", "--repeats", "1",
                        "--max-validation-samples", "64", "--output", str(output)],
                       cwd=ROOT, env=env, creationflags=flags, check=True,
                       stdout=subprocess.DEVNULL)
        print(f"Smoke passed: {output}")
        return
    status_path = OUT / "status.json"
    if args.resume:
        if not status_path.is_file():
            raise FileNotFoundError("Cannot resume without status.json")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") not in {"paused_by_user", "failed"}:
            raise ValueError(f"Refusing to resume status={status.get('status')!r}")
        status["status"] = "running"
        status["error"] = None
        status["resumed_at"] = now()
    else:
        if status_path.exists():
            raise FileExistsError("Follow-up status already exists; refusing duplicate launch")
        status = {"status": "running", "started_at": now(), "updated_at": now(),
                  "protocol": "frozen_srm_validation_only_attribution_and_isolated_repeat_v1",
                  "preflight": checked, "jobs": [], "attribution": None, "error": None}
    save(status)
    try:
        # First pair produces matched prediction arrays for attribution.
        schedule = [("PEMS08", "sparse", 1), ("PEMS08", "full", 1)]
        schedule += [(dataset, mode, repeat)
                     for dataset in CONFIGS for repeat in range(1, 6)
                     for mode in ("sparse", "full")
                     if not (dataset == "PEMS08" and repeat == 1)]
        completed_jobs = {(row["dataset"], row["mode"], row["repeat"])
                          for row in status["jobs"]}
        for dataset, mode, repeat in schedule:
            if (dataset, mode, repeat) in completed_jobs:
                continue
            print(f"{now()} begin {dataset} {mode} repeat {repeat}", flush=True)
            status["jobs"].append(run_job(dataset, mode, repeat))
            print(f"{now()} done {dataset} {mode} repeat {repeat}", flush=True)
            if len(status["jobs"]) == 2:
                sparse = OUT / "pems08_sparse_r1_predictions.npz"
                full = OUT / "pems08_full_r1_predictions.npz"
                status["attribution"] = per_node_and_day_attribution(sparse, full)
                write_json(OUT / "pems08_attribution.json", status["attribution"])
            save(status)
        status["status"] = "complete"
    except Exception:
        status["status"] = "failed"
        status["error"] = traceback.format_exc()
        raise
    finally:
        save(status)


if __name__ == "__main__":
    main()
