"""Paired full-Teacher evaluation of raw and coded retrieval containers."""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "PEMS04"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "val"
PLAN_ARG = sys.argv[3] if len(sys.argv) > 3 else None
if PLAN_ARG and PLAN_ARG.startswith("restore:"):
    rule = json.loads((HERE / "frozen_rule_v1.json").read_text(encoding="utf-8"))
    PLAN = rule["selection"][DATASET]["plan_keep"].copy()
    steps = int(PLAN_ARG.split(":", 1)[1])
    order = rule["backoff_order"]
    for cursor in range(steps):
        field = order[cursor % len(order)]
        ladder = rule["ladders"][field]
        current = next(i for i, spec in enumerate(ladder) if spec == PLAN[field])
        if current + 1 < len(ladder):
            PLAN[field] = ladder[current + 1]
else:
    PLAN = json.loads(PLAN_ARG) if PLAN_ARG and PLAN_ARG != "packaged" else None
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import container as codec
from run import trainer, unpack_dataloader_input
from utils import datasets_utils

EXPS = {
    "PEMS03": "TLAST_improved_PEMS03_12to12_exppaper_pems03_s1_final_joint",
    "PEMS04": "TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3",
    "PEMS07": "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint",
    "PEMS08": "TLAST_improved_PEMS08_12to12_exppaper_pems08_s1_final_joint",
}


def package_fields(ds, method, split):
    if method == "candidate":
        raw, _ = package_fields(ds, "raw", split)
        blobs, layout = codec.encode_fields(raw, PLAN)
        return codec.decode_fields(blobs, layout), sum(blob.nbytes for blob in blobs.values())
    path = HERE / "packages" / "bench" / f"{ds}__{method}__{split}.fcz"
    blobs, header = codec.read_package(path)
    fields = codec.decode_fields(blobs, header["layout"])
    return fields, path.stat().st_size


def main():
    if SPLIT not in ("val", "test"):
        raise ValueError(SPLIT)
    exp = ROOT / "EXP_experiments" / "TLAST_improved" / DATASET / EXPS[DATASET]
    with (exp / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.cache_data_path = "datasets"
    args.pearson_adj = None
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(v) for v in args.k_hops.split(",")]
    args.use_analog_code_domain_aggregation = 0
    args.analog_candidate_code_cache = "none"
    logger = logging.getLogger("codec_full_teacher")
    logger.addHandler(logging.NullHandler())
    data, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(exp / "best.pth", map_location=device), strict=True)
    engine.model.eval()
    loader = data[f"{SPLIT}_loader"]
    original_retrieval = loader.retrieval_data
    original_exact = loader.exact_retrieval_data
    original_available = loader.exact_available_data
    results = {}

    candidate_name = ("candidate" if PLAN is not None else
                      "teacher_restored" if PLAN_ARG == "packaged" else "ours_full")
    for method in ("raw", candidate_name):
        fields, size = package_fields(DATASET, method, SPLIT)
        real_samples = len(fields["T"])
        if real_samples > len(loader.idx):
            raise ValueError("Container has more queries than loader")
        if real_samples < len(loader.idx):
            padding = len(loader.idx) - real_samples
            fields = {key: np.concatenate((value, np.repeat(value[-1:], padding, axis=0)), axis=0)
                      for key, value in fields.items()}
        std = float(np.asarray(scaler.std).reshape(-1)[0])
        quality = np.stack((
            np.broadcast_to(fields["hd"][:, None, :] / (std * std), fields["T"].shape),
            fields["fd"] / std,
        ), axis=-1).astype(np.float32)
        loader.retrieval_data = np.concatenate((fields["T"][..., None], quality), axis=-1)
        loader.exact_retrieval_data = fields["E"][..., None].astype(np.float32)
        loader.exact_available_data = fields["B"].astype(np.uint8)
        total_abs = 0.0
        total_count = 0
        hz_abs = np.zeros(args.predict_length, dtype=np.float64)
        hz_count = np.zeros(args.predict_length, dtype=np.int64)
        processed = 0
        with torch.no_grad():
            for loader_input, loader_target in loader.get_iterator():
                x, xt, ar, er, ea, cf, ca, cfeat = unpack_dataloader_input(loader_input)
                y, yt = loader_target
                model_input = [torch.as_tensor(x, dtype=torch.float32, device=device),
                               torch.as_tensor(xt, dtype=torch.long, device=device),
                               torch.as_tensor(yt, dtype=torch.long, device=device)]
                for value in (ar, er, ea, cf, ca, cfeat):
                    if value is not None:
                        model_input.append(torch.as_tensor(value, dtype=torch.float32, device=device))
                keep = min(len(y), real_samples - processed)
                prediction = engine._forward(model_input)[:keep]
                target = torch.as_tensor(y[:keep], dtype=torch.float32, device=device)
                valid = torch.isfinite(target) & (target != 0)
                error = (prediction - target).abs()
                total_abs += float(error[valid].sum().cpu())
                total_count += int(valid.sum().cpu())
                hz_abs += (error * valid).sum(dim=(0, 2, 3)).cpu().numpy()
                hz_count += valid.sum(dim=(0, 2, 3)).cpu().numpy()
                processed += keep
        results[method] = {"mae_global": total_abs / total_count,
                           "mae_horizon_mean": float(np.mean(hz_abs / hz_count)),
                           "bytes": size, "samples": processed}
    loader.retrieval_data = original_retrieval
    loader.exact_retrieval_data = original_exact
    loader.exact_available_data = original_available
    results["delta_mae"] = results[candidate_name]["mae_horizon_mean"] - results["raw"]["mae_horizon_mean"]
    if PLAN is not None:
        results["candidate_plan"] = PLAN
    print(json.dumps({"dataset": DATASET, "split": SPLIT, "results": results}, indent=2))


if __name__ == "__main__":
    main()
