"""Rebuild the full-retrieval Validation predictions from the frozen checkpoint."""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0]]
from run import trainer, unpack_dataloader_input
from utils import datasets_utils
from calibrate_dualscale_quantile_residual import metrics

EXPERIMENT = ROOT / "EXP_experiments/TLAST_improved/PEMSBAY/TLAST_improved_PEMSBAY_12to12_exppemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1"
OUTPUT = HERE / "full_validation_replayed.npz"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with (EXPERIMENT / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.pearson_adj = None
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("pemsbay_full_validation")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    loaders, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(EXPERIMENT / "best.pth", map_location=device,
                                            weights_only=True), strict=True)
    model = engine.model.eval()
    loader = loaders["val_loader"]
    old = (loader.drop_last, loader.num_batch)
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    prediction, truth = [], []
    try:
        with torch.inference_mode():
            for loader_input, loader_target in loader.get_iterator():
                x, x_time, *extra = unpack_dataloader_input(loader_input)
                y, y_time = loader_target
                inputs = [torch.as_tensor(x, dtype=torch.float32, device=device),
                          torch.as_tensor(x_time, dtype=torch.long, device=device),
                          torch.as_tensor(y_time, dtype=torch.long, device=device)]
                inputs.extend(torch.as_tensor(value, device=device) for value in extra if value is not None)
                output, _ = model(inputs)
                prediction.append(scaler.inverse_transform(output).cpu().numpy()[..., 0])
                truth.append(np.asarray(y)[..., 0])
    finally:
        loader.drop_last, loader.num_batch = old
    prediction = np.concatenate(prediction)[:len(loader.idx)]
    truth = np.concatenate(truth)[:len(loader.idx)]
    with np.load(ROOT / "datasets/PEMSBAY/PEMSBAY_12to12_index_scaler_712.npz") as archive:
        endpoints = archive["idx_val"]
    if len(prediction) != len(endpoints) or prediction.shape != truth.shape:
        raise ValueError("Validation shapes do not align")
    np.savez_compressed(OUTPUT, val_prediction=prediction, val_truth=truth,
                        val_endpoints=endpoints)
    print(f"Replayed full Validation: {metrics(prediction, truth)}", flush=True)


if __name__ == "__main__":
    main()
