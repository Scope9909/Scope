"""Compare the official Test call and runtime benchmark call on one batch."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATASET = sys.argv[1] if len(sys.argv) > 1 else "PEMS04"
sys.argv = [sys.argv[0]]

from run import trainer, unpack_dataloader_input
from utils import datasets_utils, utils


def main():
    ds = DATASET
    exp_names = {
        "PEMS03": "TLAST_improved_PEMS03_12to12_exppaper_pems03_s1_final_joint",
        "PEMS04": "TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3",
        "PEMS07": "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint",
        "PEMS08": "TLAST_improved_PEMS08_12to12_exppaper_pems08_s1_final_joint",
    }
    exp = ROOT / "EXP_experiments" / "TLAST_improved" / ds / exp_names[ds]
    with (exp / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.cache_data_path = "datasets"
    args.pearson_adj = None
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(v) for v in args.k_hops.split(",")]
    args.use_analog_code_domain_aggregation = 0
    args.analog_candidate_code_cache = "none"

    import logging
    logger = logging.getLogger("path_pair")
    logger.addHandler(logging.NullHandler())
    data, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(exp / "best.pth", map_location=device), strict=True)
    engine.model.eval()

    input_dtypes = {}
    max_diff = 0.0
    total_abs = 0.0
    total_count = 0
    official_horizon_abs = None
    official_horizon_count = None
    n_batches = 0
    first_batch_mae = None
    real_samples = len(np.load(ROOT / "datasets" / ds / f"{ds}_12to12_index_scaler.npz")["idx_test"])
    processed = 0
    for loader_input, loader_target in data["test_loader"].get_iterator():
        x, xt, ar, er, ea, cf, ca, cfeat = unpack_dataloader_input(loader_input)
        y, yt = loader_target
        shared = [torch.as_tensor(x, dtype=torch.float32, device=device),
                  torch.as_tensor(xt, dtype=torch.long, device=device),
                  torch.as_tensor(yt, dtype=torch.long, device=device)]
        official = list(shared)
        benchmark = list(shared)
        for name, value in (("retrieval", ar), ("exact", er), ("exact_available", ea),
                            ("candidate_future", cf), ("candidate_available", ca),
                            ("candidate_features", cfeat)):
            if value is None:
                continue
            input_dtypes[name] = str(value.dtype)
            official.append(torch.as_tensor(value, dtype=torch.float32, device=device))
            kind = torch.float16 if value.dtype == np.float16 else torch.float32
            benchmark.append(torch.as_tensor(value, dtype=kind, device=device))

        with torch.no_grad():
            official_pred = scaler.inverse_transform(engine.model(official)[0])
            benchmark_pred = engine._forward(benchmark)
        keep = min(len(y), real_samples - processed)
        official_pred = official_pred[:keep]
        benchmark_pred = benchmark_pred[:keep]
        truth = torch.as_tensor(y[:keep], dtype=torch.float32, device=device)
        valid = torch.isfinite(truth) & (truth != 0)
        max_diff = max(max_diff, float((official_pred - benchmark_pred).abs().max().cpu()))
        abs_error = (official_pred - truth).abs()
        total_abs += float(abs_error[valid].sum().cpu())
        total_count += int(valid.sum().cpu())
        horizon_abs = (abs_error * valid).sum(dim=(0, 2, 3)).cpu().numpy()
        horizon_count = valid.sum(dim=(0, 2, 3)).cpu().numpy()
        if official_horizon_abs is None:
            official_horizon_abs = horizon_abs.astype(np.float64)
            official_horizon_count = horizon_count.astype(np.int64)
        else:
            official_horizon_abs += horizon_abs
            official_horizon_count += horizon_count
        if n_batches == 0:
            first_batch_mae = float(utils.metric_torch(official_pred, truth, mask_val=args.eval_mask)[0])
        processed += keep
        n_batches += 1
    result = {
        "dataset": ds,
        "device": str(device),
        "input_dtypes": input_dtypes,
        "batches": n_batches,
        "samples": processed,
        "first_batch_mae": first_batch_mae,
        "max_prediction_difference": max_diff,
        "test_global_mae": total_abs / total_count,
        "test_horizon_mean_mae": float(np.mean(official_horizon_abs / official_horizon_count)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
