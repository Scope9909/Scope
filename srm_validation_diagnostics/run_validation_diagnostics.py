"""Frozen SRM diagnostic replay on Validation only; never reads Test indices."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
ORIGINAL_ARGV = sys.argv[1:]
sys.path.insert(0, str(ROOT / "_runtime_deps"))
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch
sys.argv = [sys.argv[0]]  # run.py has an import-time argument parser.
from run import trainer, unpack_dataloader_input  # noqa: E402
from utils import datasets_utils  # noqa: E402
from utils.online_residual_memory import OnlineResidualMemory  # noqa: E402

CONFIGS = {
    "PEMS03": {
        "experiment": "TLAST_improved_PEMS03_12to12_exp10101500_s1",
        "state_dir": "pems03_int8_uniform",
        "state_pattern": "{mode}_cold_state.npz",
        "selector": "first55_three_block_stability_int8_topk36_budget",
    },
    "PEMS04": {
        "experiment": "TLAST_improved_PEMS04_12to12_exppems04_joint3_lite_r256_strictalign_s1",
        "state_dir": "cold_start_streaming",
        "state_pattern": "pems04_{mode}_cold_state.npz",
        "selector": "actual_online_state_byte_capped_int8_s1",
    },
    "PEMS07": {
        "experiment": "TLAST_improved_PEMS07_12to12_exppems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1",
        "state_dir": "cold_start_streaming",
        "state_pattern": "pems07_{mode}_cold_state.npz",
        "selector": "first55_three_block_stability_int8_s1",
    },
    "PEMS08": {
        "experiment": "TLAST_improved_PEMS08_12to12_exppems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1",
        "state_dir": "cold_start_streaming",
        "state_pattern": "pems08_{mode}_cold_state.npz",
        "selector": "actual_online_state_byte_capped_int8_s1",
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def paths(dataset: str) -> tuple[Path, Path, Path, Path]:
    cfg = CONFIGS[dataset]
    exp = ROOT / "EXP_experiments" / "TLAST_improved" / dataset / cfg["experiment"]
    states = ROOT / cfg["state_dir"]
    sparse = states / cfg["state_pattern"].format(mode="sparse")
    full = states / cfg["state_pattern"].format(mode="full")
    index = ROOT / "datasets" / dataset / f"{dataset}_12to12_index_scaler.npz"
    for path in (exp / "config.pkl", exp / "best.pth", sparse, full, index):
        if not path.is_file():
            raise FileNotFoundError(path)
    return exp, sparse, full, index


def preflight() -> dict:
    result = {}
    for dataset in CONFIGS:
        exp, sparse_path, full_path, index_path = paths(dataset)
        with np.load(index_path, allow_pickle=False) as archive:
            endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
        sparse = OnlineResidualMemory.from_npz(sparse_path)
        full = OnlineResidualMemory.from_npz(full_path)
        if sparse.retrieval_format != "int8" or full.retrieval_format != "int8":
            raise ValueError(f"{dataset}: expected identical INT8 retrieval format")
        if not np.isin(sparse.node_ids, full.node_ids).all():
            raise ValueError(f"{dataset}: sparse nodes are not a subset of full")
        if sparse.cold_start_endpoint != int(endpoints[0]):
            raise ValueError(f"{dataset}: cold-start endpoint does not match Validation")
        if not np.all(np.diff(endpoints) > 0):
            raise ValueError(f"{dataset}: Validation endpoints are not ordered")
        result[dataset] = {
            "checkpoint": str(exp / "best.pth"),
            "sparse_state": str(sparse_path), "full_state": str(full_path),
            "validation_samples": len(endpoints),
            "selected_nodes": len(sparse.node_ids), "all_nodes": len(full.node_ids),
            "selector": CONFIGS[dataset]["selector"],
        }
    return result


def sparse_correction(model, prediction, ids, retrieval, exact, available):
    low = model.analog_lowrank_retrieval_residual
    adapter = model.analog_exact_match_residual
    selected = prediction.index_select(2, ids)
    logits = low.horizon_logits[:, None] + torch.einsum(
        "nr,hr->hn", low.node_factor.index_select(0, ids), low.horizon_factor)
    blend = low.max_blend * torch.tanh(logits).view(1, logits.shape[0], len(ids), 1)
    selected = selected + blend * (retrieval - selected)
    exact_blend = (adapter.max_blend * torch.tanh(adapter.horizon_logits)).view(1, -1, 1, 1)
    selected = selected + exact_blend * available.to(selected.dtype) * (exact - selected)
    return prediction.index_copy(2, ids, selected)


def array_payload_bytes(memory: OnlineResidualMemory) -> int:
    names = ("train_raw", "idx_train", "train_tod", "pending_idx_train",
             "pending_train_tod", "node_ids", "quant_scale", "scaler_mean", "scaler_std")
    return int(sum(getattr(memory, name).nbytes for name in names))


def fold_of(position: np.ndarray, n: int) -> np.ndarray:
    edges = [int(n * f) for f in (0.55, 0.70, 0.85, 1.0)]
    return np.searchsorted(edges, position, side="right") - 1


def update_stats(stats: dict, prediction: np.ndarray, truth: np.ndarray,
                 target_tod: np.ndarray, selected_mask: np.ndarray,
                 positions: np.ndarray, n: int) -> None:
    # Micro-MAE with the repository's round(target)>0 validity mask.
    error = np.abs(prediction.astype(np.float64) - truth.astype(np.float64))
    valid = np.round(truth) > 0
    peak = ((target_tod >= 84) & (target_tod < 108)) | ((target_tod >= 204) & (target_tod < 228))
    folds = fold_of(positions, n)
    for fold in range(3):
        in_fold = folds == fold
        if not in_fold.any():
            continue
        for group, node_mask in (("all", np.ones_like(selected_mask)),
                                 ("selected", selected_mask),
                                 ("unselected", ~selected_mask)):
            if not node_mask.any():
                continue
            for period, time_mask in (("all", np.ones_like(peak, dtype=bool)),
                                      ("peak", peak), ("offpeak", ~peak)):
                for horizon, start, end in (("all", 0, 12), ("h01_03", 0, 3),
                                            ("h04_06", 3, 6), ("h07_09", 6, 9),
                                            ("h10_12", 9, 12)):
                    mask = (valid[in_fold, start:end, :]
                            & time_mask[in_fold, start:end, None]
                            & node_mask[None, None, :])
                    count = int(mask.sum())
                    if count:
                        key = f"fold{fold + 1}/{group}/{period}/{horizon}"
                        stats[key][0] += float(error[in_fold, start:end, :][mask].sum())
                        stats[key][1] += count


def run_dataset(dataset: str, process: psutil.Process) -> dict:
    exp, sparse_path, full_path, index_path = paths(dataset)
    with (exp / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.pearson_adj = None
    if not (args.use_analog_lowrank_retrieval_residual and
            args.use_analog_exact_match_residual):
        raise ValueError(f"{dataset}: unexpected checkpoint architecture")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("srm_validation_diagnostics")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(exp / "best.pth", map_location=device,
                                             weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False
    with np.load(index_path, allow_pickle=False) as archive:
        endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
    sparse_ids = OnlineResidualMemory.from_npz(sparse_path).node_ids
    selected_mask = np.zeros(nodes, dtype=bool)
    selected_mask[sparse_ids] = True
    loader = loaders["val_loader"]
    old_drop, old_batches, old_bs = loader.drop_last, loader.num_batch, loader.bs
    loader.bs = 32
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    modes = {}
    try:
        for mode, state_path in (("backbone", None), ("sparse", sparse_path),
                                 ("full", full_path)):
            memory = OnlineResidualMemory.from_npz(state_path) if state_path else None
            ids = (torch.as_tensor(memory.node_ids, dtype=torch.long, device=device)
                   if memory is not None else None)
            stats = defaultdict(lambda: [0.0, 0])
            start_payload = array_payload_bytes(memory) if memory else 0
            start_candidates = len(memory.idx_train) if memory else 0
            update_ns = [0]
            update_calls = [0]
            activated = [0]
            observed_frames = [0]
            if memory is not None:
                original_update = memory._observe_and_activate

                def timed_update(endpoint, history, tod=None):
                    before_candidates = len(memory.idx_train)
                    before_frames = len(memory.train_raw)
                    tic = time.perf_counter_ns()
                    try:
                        return original_update(endpoint, history, tod)
                    finally:
                        update_ns[0] += time.perf_counter_ns() - tic
                        update_calls[0] += 1
                        activated[0] += len(memory.idx_train) - before_candidates
                        observed_frames[0] += len(memory.train_raw) - before_frames

                memory._observe_and_activate = timed_update
            cursor = 0
            query_ns = 0
            peak_rss = process.memory_info().rss
            print(f"{now()} {dataset} {mode} begin", flush=True)
            with torch.inference_mode():
                for loader_input, loader_target in loader.get_iterator():
                    if cursor >= len(endpoints):
                        break
                    x, x_time, *extras = unpack_dataloader_input(loader_input)
                    if any(value is not None for value in extras):
                        raise ValueError("Precomputed query cache unexpectedly loaded")
                    y, y_time = loader_target
                    take = min(len(x), len(endpoints) - cursor)
                    x, x_time, y, y_time = x[:take], x_time[:take], y[:take], y_time[:take]
                    if memory is not None:
                        raw_history = np.rint(
                            scaler.inverse_transform(x)[..., 0][:, :, memory.node_ids]
                        ).astype(np.float32)
                        tod = np.asarray(x_time[:, -1, 0], dtype=np.int16)
                        tic = time.perf_counter_ns()
                        retrieval, exact, available, _ = memory.query_stream(
                            raw_history, endpoints[cursor:cursor + take], tod)
                        query_ns += time.perf_counter_ns() - tic
                    model_input = [
                        torch.as_tensor(x, dtype=torch.float32, device=device),
                        torch.as_tensor(x_time, dtype=torch.long, device=device),
                        torch.as_tensor(y_time, dtype=torch.long, device=device),
                    ]
                    prediction, _ = model(model_input)
                    if memory is not None:
                        prediction = sparse_correction(
                            model, prediction, ids,
                            torch.as_tensor(retrieval, device=device),
                            torch.as_tensor(exact, device=device).float(),
                            torch.as_tensor(available, device=device))
                    pred = scaler.inverse_transform(prediction).cpu().numpy()[..., 0]
                    truth = np.asarray(y)[..., 0]
                    target_tod = np.asarray(y_time)[..., 0]
                    update_stats(stats, pred, truth, target_tod, selected_mask,
                                 np.arange(cursor, cursor + take), len(endpoints))
                    cursor += take
                    peak_rss = max(peak_rss, process.memory_info().rss)
            if cursor != len(endpoints):
                raise ValueError(f"{dataset}/{mode}: Validation loader covered {cursor}/{len(endpoints)}")
            modes[mode] = {
                "stats": {key: {"absolute_error_sum": value[0], "valid_count": value[1],
                                "mae": value[0] / value[1]} for key, value in stats.items()},
                "state_file_bytes": state_path.stat().st_size if state_path else 0,
                "array_payload_start_bytes": start_payload,
                "array_payload_end_bytes": array_payload_bytes(memory) if memory else 0,
                "active_candidates_start": start_candidates,
                "active_candidates_end": len(memory.idx_train) if memory else 0,
                "activated_candidates": activated[0],
                "observed_frames_appended": observed_frames[0],
                "pending_end": memory.pending_count if memory else 0,
                "maintenance_calls": update_calls[0],
                "maintenance_seconds_total": update_ns[0] / 1e9,
                "retrieval_seconds_total": query_ns / 1e9,
                "process_rss_peak_sampled_bytes": peak_rss,
            }
            print(f"{now()} {dataset} {mode} done", flush=True)
    finally:
        loader.drop_last, loader.num_batch, loader.bs = old_drop, old_batches, old_bs
    return {"dataset": dataset, "seed": 1, "validation_samples": len(endpoints),
            "nodes": nodes, "selected_nodes": len(sparse_ids),
            "selector": CONFIGS[dataset]["selector"], "modes": modes}


def value(row: dict, mode: str, fold: int, group: str = "all",
          period: str = "all", horizon: str = "all") -> float:
    key = f"fold{fold}/{group}/{period}/{horizon}"
    return row["modes"][mode]["stats"][key]["mae"]


def render_report(status: dict) -> str:
    lines = [
        "# SRM Validation 分组误差与在线维护成本（2026-10-15）", "",
        f"状态：**{status['status']}**；开始：{status['started_at']}；最后更新：{status['updated_at']}。", "",
        "## 论文口径", "",
        "1. 贡献一：以既有 ICASSP 检索增强方法为原始预测器与问题基础；期刊/会议扩展需清楚标出继承部分与新增部分，不能把既有方法原封不动算作新的独立创新。",
        "2. 贡献二：DCLD 的低秩 Student、严格对齐的蒸馏与 Validation-only 校准；参数量与完整系统成本分别报告。",
        "3. 贡献三：SRM（节点级稀疏残差记忆）。冻结节点集合后，只在选中节点保留 Train-only 在线记忆并施加检索残差；未选节点使用同一 Student 的 backbone。主张可部署机制及精度—资源权衡，不主张选点公式为已充分验证的最优分配算法。", "",
        "本报告是追加的诊断证据，不改变既有冻结规则，也不替换主表。PEMS07 曾参与选点规则设计；PEMS03/08 的既有三个 seed 标签不代表独立 checkpoint 方差。", "",
        "## 协议", "",
        "- 仅使用官方 Validation `idx_val` 和 seed=1 已有 checkpoint/冷启动 INT8 状态；**不读取 Test index、Test label，不重选节点，不拟合新校准器**。", "",
        "- 后续折按 Validation 样本位置分为 55–70%、70–85%、85–100%；报告未校准 raw MAE，作为机制诊断，不与已发表的 Validation 校准 MAE 混同。", "",
        "- 有效标签沿用仓库约定 `round(y)>0`；分组 MAE 是有效元素的 micro-average。峰时按目标时刻 07:00–09:00、17:00–19:00（5 分钟槽），其余为非峰时；这只是预设解释性切片。", "",
        "- 在线维护时间只计 `query_stream` 中 `_observe_and_activate` 的累计墙钟时间；检索时间包括维护。数组载荷字节只含显式 NumPy 数组，不代表进程总 RAM。采样 RSS 峰值含模型和数据集，并非纯记忆峰值。", "",
        "## 结果", "",
    ]
    if not status["results"]:
        lines.append("后台任务尚未完成首个数据集。")
    for dataset, row in status["results"].items():
        lines += [f"### {dataset}（seed=1；{row['selected_nodes']}/{row['nodes']} 节点）", "",
                  "| 后续 Validation 折 | Sparse raw MAE | Full raw MAE | Sparse−Full | Selected 节点差 | Unselected 节点差 |", "|---|---:|---:|---:|---:|---:|"]
        for fold in (1, 2, 3):
            sparse = value(row, "sparse", fold)
            full = value(row, "full", fold)
            sel = value(row, "sparse", fold, "selected") - value(row, "full", fold, "selected")
            unsel = value(row, "sparse", fold, "unselected") - value(row, "full", fold, "unselected")
            lines.append(f"| {fold} | {sparse:.6f} | {full:.6f} | {sparse-full:+.6f} | {sel:+.6f} | {unsel:+.6f} |")
        lines += ["", "选中/未选中节点的完整 peak/offpeak × 4 个预测步长分组保存在同目录 `results.json`；下表汇总稀疏相对全记忆的峰时与长步长差。", "",
                  "| 折 | Peak Sparse−Full | Offpeak Sparse−Full | H01–03 Sparse−Full | H10–12 Sparse−Full |", "|---|---:|---:|---:|---:|"]
        for fold in (1, 2, 3):
            diffs = [value(row, "sparse", fold, period=p, horizon=h) -
                     value(row, "full", fold, period=p, horizon=h)
                     for p, h in (("peak", "all"), ("offpeak", "all"),
                                  ("all", "h01_03"), ("all", "h10_12"))]
            lines.append("| {} | {} | {} | {} | {} |".format(
                fold, *(f"{diff:+.6f}" for diff in diffs)))
        lines += ["", "| 在线模式 | 冷启动状态文件 | 显式数组载荷起→止 | 释放候选数 | 追加观测帧 | 维护总时长 | 检索总时长 | 采样进程 RSS 峰值 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for mode in ("sparse", "full"):
            m = row["modes"][mode]
            mib = 1024 ** 2
            lines.append(f"| {mode} | {m['state_file_bytes']/mib:.3f} MiB | "
                         f"{m['array_payload_start_bytes']/mib:.3f}→{m['array_payload_end_bytes']/mib:.3f} MiB | "
                         f"{m['activated_candidates']} | {m['observed_frames_appended']} | "
                         f"{m['maintenance_seconds_total']:.4f} s | "
                         f"{m['retrieval_seconds_total']:.2f} s | "
                         f"{m['process_rss_peak_sampled_bytes']/mib:.1f} MiB |")
        lines += ["", "注意：这是一次顺序回放，不是重复延迟分布或并发吞吐；采样 RSS 只用于工程诊断，不能当作算法纯内存节省。", ""]
    if status.get("error"):
        lines += ["## 运行错误", "", "```text", status["error"], "```", ""]
    lines += ["## 来源与限制", "",
              "- 冻结 Test-once 汇总：`../srm_test_once/FROZEN_SRM_TEST_ONCE_REPORT.md`。本轮不复算 Test。",
              "- 已有三 seed Validation：`../cold_start_single_request_multiseed/FOUR_DATASET_THREE_SEED_VALIDATION.md`。本轮分组结果只做 seed=1，不冒充三 seed 独立复现。",
              "- 若峰时、节点或步长切片显示 PEMS08 的误差集中，只能解释当前冻结方案的代价，不能在看过本轮切片后调选点并把同一 Validation 当独立验证。", ""]
    return "\n".join(lines)


def save(status: dict) -> None:
    status["updated_at"] = now()
    write_json(OUT / "results.json", status)
    temp = OUT / "SRM_VALIDATION_DIAGNOSTICS_20261015.md.tmp"
    temp.write_text(render_report(status), encoding="utf-8")
    temp.replace(OUT / "SRM_VALIDATION_DIAGNOSTICS_20261015.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    # sys.argv was cleared for run.py; parse the saved invocation instead.
    args = parser.parse_args(ORIGINAL_ARGV)
    checked = preflight()
    if args.preflight:
        print(json.dumps(checked, ensure_ascii=False, indent=2))
        return
    torch.set_num_threads(min(4, torch.get_num_threads()))
    status = {"status": "running", "started_at": now(), "updated_at": now(),
              "protocol": "frozen_srm_validation_only_seed1_diagnostic_v1",
              "preflight": checked, "results": {}, "error": None}
    save(status)
    process = psutil.Process()
    try:
        for dataset in CONFIGS:
            status["results"][dataset] = run_dataset(dataset, process)
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
