"""Read-only paired Validation audit of the PEMS07 four-neighbor best checkpoint.

The Teacher--Student top-20 group is frozen from the *original* Student and
Teacher predictions; candidate outcomes are never used to select that group.
No Test iterator or Test prediction key is accessed.
"""

from __future__ import annotations

import gc
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from contrib3_horizon_conflict_20261016.audit_horizon_gradients import tensorize_model_input
from models import TLAST_improved
from utils import datasets_utils


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "EXP_experiments/TLAST_improved/PEMS07"
BASE_EXP = EXP_ROOT / "TLAST_improved_PEMS07_12to12_exppems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1"
NEIGHBOR_EXP = EXP_ROOT / "TLAST_improved_PEMS07_12to12_expcontrib3_pems07_neighbor_k4_s1"
BASE_CACHE = ROOT / "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz"
TEACHER_CACHE = ROOT / "datasets/PEMS07/PEMS07_12to12_parent_predictions_joint3_strict.npz"


def mae(prediction: np.ndarray, truth: np.ndarray, nodes=None, rows=None) -> float:
    if rows is None:
        rows = slice(None)
    if nodes is None:
        nodes = slice(None)
    p = prediction[rows, :, nodes]
    y = truth[rows, :, nodes]
    valid = np.isfinite(y) & (np.abs(y) >= 1e-4)
    return float(np.abs(p - y)[valid].astype(np.float64).mean())


def predict(exp: Path, args, loader, scaler, device) -> tuple[np.ndarray, np.ndarray]:
    model = TLAST_improved.Model(args).to(device)
    state = torch.load(exp / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    predictions, truths = [], []
    with torch.inference_mode():
        for batch_input, (y, y_time) in loader.get_iterator():
            model_input = tensorize_model_input(batch_input, y_time, device)
            output, _ = model(model_input)
            predictions.append(scaler.inverse_transform(output)[..., 0].cpu().numpy())
            truths.append(np.asarray(y[..., 0], dtype=np.float32).copy())
    del model, state
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(predictions), np.concatenate(truths)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("paired-validation")
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with (BASE_EXP / "config.pkl").open("rb") as stream:
        base_args = pickle.load(stream)
    with (NEIGHBOR_EXP / "config.pkl").open("rb") as stream:
        neighbor_args = pickle.load(stream)
    for key in ("dataset_name", "input_length", "predict_length", "in_dim", "pre_dim", "train_ratio", "test_ratio", "analog_retrieval_cache", "batch_size", "eval_mask"):
        if getattr(base_args, key) != getattr(neighbor_args, key):
            raise ValueError(f"Baseline/candidate data mismatch at {key}")
    if int(base_args.eval_mask) != 0:
        raise ValueError("Audit expects zero-masked MAE")

    # The loader constructor initializes all split objects, but only the
    # Validation iterator below is traversed; no Test metric is computed.
    loaders, num_nodes, scaler, _ = datasets_utils.load_dataset(base_args, logger)
    loader = loaders["val_loader"]
    for args in (base_args, neighbor_args):
        args.num_nodes = num_nodes
        args.pearson_adj = None

    base, truth = predict(BASE_EXP, base_args, loader, scaler, device)
    neighbor, neighbor_truth = predict(NEIGHBOR_EXP, neighbor_args, loader, scaler, device)
    if not np.array_equal(truth, neighbor_truth):
        raise ValueError("The two checkpoints were not evaluated on identical Validation labels")
    if base.shape != (len(loader.idx), 12, num_nodes):
        raise ValueError(f"Unexpected prediction shape {base.shape}")

    with np.load(BASE_CACHE, allow_pickle=False) as cache:
        if not np.array_equal(loader.idx, cache["val_endpoints"]):
            raise ValueError("Validation endpoint alignment with original cache failed")
        if not np.array_equal(truth, cache["val_truth"]):
            raise ValueError("Validation truth alignment with original cache failed")
        cached_base_mae = mae(cache["val_prediction"], truth)
        maximum_base_cache_difference = float(np.max(np.abs(base - cache["val_prediction"])))
        mean_base_cache_difference = float(np.mean(np.abs(base - cache["val_prediction"]), dtype=np.float64))
        base_cache_mae_difference = abs(mae(base, truth) - cached_base_mae)
        if mean_base_cache_difference > 1e-3 or base_cache_mae_difference > 1e-3:
            raise ValueError(
                "Best-checkpoint baseline does not reproduce original cache: "
                f"max prediction difference={maximum_base_cache_difference}, "
                f"mean prediction difference={mean_base_cache_difference}, "
                f"MAE difference={base_cache_mae_difference}"
            )

    with np.load(TEACHER_CACHE, allow_pickle=False) as cache:
        teacher_normalized = cache["teacher_pred_val"][..., 0]
        mean = float(cache["scaler_mean"])
        std = float(cache["scaler_std"])
    teacher = teacher_normalized * std + mean
    valid = np.isfinite(truth) & (np.abs(truth) >= 1e-4)
    base_abs = np.abs(base - truth)
    neighbor_abs = np.abs(neighbor - truth)
    teacher_abs = np.abs(teacher - truth)
    node_count = valid.sum(axis=(0, 1))
    node_baseline = np.where(valid, base_abs, 0).sum(axis=(0, 1), dtype=np.float64) / node_count
    node_neighbor = np.where(valid, neighbor_abs, 0).sum(axis=(0, 1), dtype=np.float64) / node_count
    node_teacher = np.where(valid, teacher_abs, 0).sum(axis=(0, 1), dtype=np.float64) / node_count
    teacher_advantage = node_baseline - node_teacher
    top20 = np.argsort(-teacher_advantage)[:20].tolist()
    rest = np.setdiff1d(np.arange(num_nodes), np.asarray(top20), assume_unique=True)
    node_gain = node_baseline - node_neighbor
    gain = mae(base, truth) - mae(neighbor, truth)

    blocks = []
    for i, indices in enumerate(np.array_split(np.arange(len(truth)), 8), start=1):
        a = mae(base, truth, rows=indices)
        b = mae(neighbor, truth, rows=indices)
        blocks.append({"block": i, "n_windows": len(indices), "baseline_mae": a, "neighbor_mae": b, "gain": a - b})

    def group_result(nodes):
        a, b = mae(base, truth, nodes=nodes), mae(neighbor, truth, nodes=nodes)
        return {"baseline_mae": a, "neighbor_mae": b, "gain": a - b, "valid_count": int(valid[:, :, nodes].sum())}

    def ranked(indices):
        return [{"node": int(i), "gain": float(node_gain[i]), "baseline_mae": float(node_baseline[i]), "neighbor_mae": float(node_neighbor[i])} for i in indices]

    total_error_reduction = float(np.where(valid, base_abs - neighbor_abs, 0).sum(dtype=np.float64))
    top20_error_reduction = float(np.where(valid[:, :, top20], base_abs[:, :, top20] - neighbor_abs[:, :, top20], 0).sum(dtype=np.float64))
    result = {
        "baseline_checkpoint": str(BASE_EXP / "best.pth"),
        "neighbor_checkpoint": str(NEIGHBOR_EXP / "best.pth"),
        "n_windows": len(truth), "n_nodes": num_nodes, "eval_mask": 0,
        "baseline_mae": mae(base, truth), "neighbor_mae": mae(neighbor, truth), "gain": gain,
        "baseline_cached_mae": cached_base_mae,
        "max_baseline_checkpoint_cache_difference": maximum_base_cache_difference,
        "mean_baseline_checkpoint_cache_difference": mean_base_cache_difference,
        "improved_nodes": int((node_gain > 0).sum()),
        "worsened_nodes": int((node_gain < 0).sum()),
        "eight_blocks": blocks,
        "improved_blocks": sum(block["gain"] > 0 for block in blocks),
        "teacher_advantage_top20_node_ids_zero_based": top20,
        "teacher_advantage_top20": group_result(top20),
        "other_863_nodes": group_result(rest),
        "top20_share_of_net_error_reduction": top20_error_reduction / total_error_reduction if total_error_reduction else None,
        "top10_candidate_improved_nodes": ranked(np.argsort(-node_gain)[:10]),
        "top10_candidate_worsened_nodes": ranked(np.argsort(node_gain)[:10]),
        "top20_teacher_nodes_candidate_gain": ranked(np.asarray(top20)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
