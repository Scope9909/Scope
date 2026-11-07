import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


ARCHIVES = {
    "PEMS03": [
        (
            ROOT / f"paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{s}_network_only/selected_predictions.npz",
            ROOT / f"paper_results/pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{s}_valcal/selected_predictions.npz",
        )
        for s in (1, 2, 3)
    ],
    "PEMS04": [
        (
            ROOT / "paper_results/lite_r256_kdfix3_a30_lr1e4_aligned_s1/selected_predictions.npz",
            ROOT / "paper_results/lite_r256_kdfix3_a30_lr1e4_smooth_tod_s1/selected_predictions.npz",
        ),
        (
            ROOT / "paper_results/lite_r256_kdfix3_a30_lr1e4_fixedparent_aligned_s2/selected_predictions.npz",
            ROOT / "paper_results/lite_r256_kdfix3_a30_lr1e4_fixedparent_smooth_tod_s2/selected_predictions.npz",
        ),
        (
            ROOT / "paper_results/lite_r256_kdfix3_a30_lr1e4_fixedparent_aligned_s3/selected_predictions.npz",
            ROOT / "paper_results/lite_r256_kdfix3_a30_lr1e4_fixedparent_smooth_tod_s3/selected_predictions.npz",
        ),
    ],
    "PEMS07": [
        (
            ROOT / f"paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{s}_network_only/selected_predictions.npz",
            ROOT / f"paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{s}_valcal/selected_predictions.npz",
        )
        for s in (1, 2, 3)
    ],
    "PEMS08": [
        (
            ROOT / f"paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{s}_network_only/selected_predictions.npz",
            ROOT / f"paper_results/pems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{s}_valcal/selected_predictions.npz",
        )
        for s in (1, 2, 3)
    ],
    "PEMSBAY": [
        (
            ROOT / "paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz",
            ROOT / "paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/selected_predictions.npz",
        )
    ],
}


def load_pair(network_path: Path, calibrated_path: Path):
    with np.load(network_path, allow_pickle=False) as net, np.load(
        calibrated_path, allow_pickle=False
    ) as cal:
        net_pred = np.asarray(net["test_prediction"], dtype=np.float64)
        cal_pred = np.asarray(cal["test_prediction"], dtype=np.float64)
        truth = np.asarray(net["test_truth"], dtype=np.float64)
        cal_truth = np.asarray(cal["test_truth"], dtype=np.float64)
        endpoints = np.asarray(net["test_endpoints"])
        cal_endpoints = np.asarray(cal["test_endpoints"])
    if net_pred.shape != cal_pred.shape or net_pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {network_path} vs {calibrated_path}")
    if not np.array_equal(truth, cal_truth):
        raise ValueError(f"truth mismatch: {network_path} vs {calibrated_path}")
    if not np.array_equal(endpoints, cal_endpoints):
        raise ValueError(f"endpoint mismatch: {network_path} vs {calibrated_path}")
    return net_pred, cal_pred, truth


def horizon_mae(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    out = []
    for horizon in range(truth.shape[1]):
        mask = truth[:, horizon] != 0
        out.append(np.abs(prediction[:, horizon] - truth[:, horizon])[mask].mean())
    return np.asarray(out, dtype=np.float64)


def sample_horizon_sums(prediction: np.ndarray, truth: np.ndarray):
    mask = truth != 0
    absolute_error = np.abs(prediction - truth) * mask
    reduce_axes = tuple(range(2, truth.ndim))
    return absolute_error.sum(axis=reduce_axes), mask.sum(axis=reduce_axes)


def paired_bootstrap(
    network_prediction: np.ndarray,
    calibrated_prediction: np.ndarray,
    truth: np.ndarray,
    repeats: int,
    seed: int,
):
    network_sum, counts = sample_horizon_sums(network_prediction, truth)
    calibrated_sum, calibrated_counts = sample_horizon_sums(calibrated_prediction, truth)
    if not np.array_equal(counts, calibrated_counts):
        raise AssertionError("paired metric masks differ")
    n = len(truth)
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    batch = 100
    for start in range(0, repeats, batch):
        size = min(batch, repeats - start)
        indices = rng.integers(0, n, size=(size, n), endpoint=False)
        count_boot = counts[indices].sum(axis=1)
        network_boot = (network_sum[indices].sum(axis=1) / count_boot).mean(axis=1)
        calibrated_boot = (calibrated_sum[indices].sum(axis=1) / count_boot).mean(axis=1)
        differences[start : start + size] = network_boot - calibrated_boot
    observed = float(horizon_mae(network_prediction, truth).mean() - horizon_mae(calibrated_prediction, truth).mean())
    return {
        "definition": "network_only_MAE_minus_valcal_MAE; positive favors ValCal",
        "unit": "test sample endpoint resampling; official horizon-mean masked MAE",
        "repeats": repeats,
        "rng_seed": seed,
        "observed_difference": observed,
        "ci95_percentile": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "probability_difference_le_zero": float(np.mean(differences <= 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "protocol": "icde_core_five_dataset_horizon_and_paired_bootstrap_v2",
        "mask": "truth != 0",
        "aggregation": "masked MAE per horizon, then arithmetic mean across 12 horizons",
        "datasets": {},
    }
    csv_rows = []
    plot_payload = {}

    for dataset, pairs in ARCHIVES.items():
        network_by_seed = []
        calibrated_by_seed = []
        first_pair = None
        for seed, (network_path, calibrated_path) in enumerate(pairs, start=1):
            network_prediction, calibrated_prediction, truth = load_pair(
                network_path, calibrated_path
            )
            if first_pair is None:
                first_pair = (network_prediction, calibrated_prediction, truth)
            network_horizon = horizon_mae(network_prediction, truth)
            calibrated_horizon = horizon_mae(calibrated_prediction, truth)
            network_by_seed.append(network_horizon)
            calibrated_by_seed.append(calibrated_horizon)
            for horizon in range(12):
                csv_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "horizon": horizon + 1,
                        "network_only_mae": network_horizon[horizon],
                        "valcal_mae": calibrated_horizon[horizon],
                        "improvement": network_horizon[horizon]
                        - calibrated_horizon[horizon],
                    }
                )
        network_by_seed = np.stack(network_by_seed)
        calibrated_by_seed = np.stack(calibrated_by_seed)
        network_mean = network_by_seed.mean(axis=0)
        calibrated_mean = calibrated_by_seed.mean(axis=0)
        ddof = 1 if len(pairs) > 1 else 0
        network_std = network_by_seed.std(axis=0, ddof=ddof)
        calibrated_std = calibrated_by_seed.std(axis=0, ddof=ddof)
        bootstrap = paired_bootstrap(
            *first_pair,
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed,
        )
        summary["datasets"][dataset] = {
            "student_seeds": len(pairs),
            "network_only_horizon_mean": network_mean.tolist(),
            "network_only_horizon_sample_std": network_std.tolist(),
            "valcal_horizon_mean": calibrated_mean.tolist(),
            "valcal_horizon_sample_std": calibrated_std.tolist(),
            "improvement_by_horizon": (network_mean - calibrated_mean).tolist(),
            "network_only_overall_horizon_mean": float(network_mean.mean()),
            "valcal_overall_horizon_mean": float(calibrated_mean.mean()),
            "paired_bootstrap_seed1": bootstrap,
        }
        plot_payload[dataset] = (network_mean, calibrated_mean)

    with (output_dir / "horizon_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (dataset, (network_mean, calibrated_mean)) in zip(axes.flat, plot_payload.items()):
        horizons = np.arange(1, 13)
        axis.plot(horizons, network_mean, marker="o", label="Lite-R256")
        axis.plot(horizons, calibrated_mean, marker="s", label="Lite-R256 + ValCal")
        axis.set_title(dataset)
        axis.set_xlabel("Forecast horizon")
        axis.set_ylabel("MAE")
        axis.set_xticks(horizons)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False)
    fig.savefig(output_dir / "horizon_mae.png", dpi=240)
    fig.savefig(output_dir / "horizon_mae.pdf")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
