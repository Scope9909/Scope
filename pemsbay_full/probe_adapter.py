"""Read-only checkpoint/old-cache adapter arithmetic at a discrepant point."""
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
checkpoint = root / "EXP_experiments/Scope/PEMSBAY/Scope_PEMSBAY_12to12_exppemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1/best.pth"
weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
print([key for key in weights if "analog_lowrank_retrieval_residual" in key or "analog_exact_match_residual" in key])
prefix = "analog_lowrank_retrieval_residual."
exact_prefix = "analog_exact_match_residual."
with np.load(root / "paper_results/pemsbay712_srm_validation_pilot/no_memory_validation_predictions.npz") as archive:
    base = archive["val_prediction"]
with np.load(root / "paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz") as archive:
    full = archive["val_prediction"]
with np.load(root / "pemsbay_full/validation_parity_debug.npz") as archive:
    sparse = archive["prediction"]
with np.load(root / "datasets/PEMSBAY/PEMSBAY_12to12_analog_k8_exact_mean_712.npz") as archive:
    retrieved = archive["retrieval_val"]
    exact = archive["exact_retrieval_val"]
    available = archive["exact_available_val"]
    mean = float(archive["scaler_mean"])
    std = float(archive["scaler_std"])

for sample, lead, node in ((4864, 12, 17), (12, 1, 0), (1024, 6, 17)):
    h = lead - 1
    b = float(base[sample, h, node])
    f = float(full[sample, h, node])
    s = float(sparse[sample, h, node])
    r = float(retrieved[sample, h, node, 0])
    e = float(exact[sample, h, node, 0])
    a = int(available[sample, h, node, 0])
    low_logit = (weights[prefix + "horizon_logits"][h] +
                 torch.dot(weights[prefix + "node_factor"][node],
                           weights[prefix + "horizon_factor"][h]))
    low_blend = float(torch.tanh(low_logit))
    ex_blend = float(torch.tanh(weights[exact_prefix + "horizon_logits"][h]))
    low_value = b + std * low_blend * (r - ((b - mean) / std))
    calc = low_value + std * ex_blend * a * (e - ((low_value - mean) / std))
    print(sample, lead, node, "base/full/sparse", b, f, s,
          "retrieved/exact/available", r, e, a,
          "blends", low_blend, ex_blend, "calculated", calc)
