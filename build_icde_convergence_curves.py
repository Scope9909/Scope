"""Build convergence curves and timing summaries from immutable logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent

RUNS = {
    "PEMS04": {
        "Teacher": [
            "EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3",
        ],
        "Lite-R256": [
            "EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_explite_r256_kdfix3_a30_lr1e4_val_s1",
            "EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_explite_r256_kdfix3_a30_lr1e4_fixedparent_s2",
            "EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_explite_r256_kdfix3_a30_lr1e4_fixedparent_s3",
        ],
    },
    "PEMS07": {
        "Teacher": [
            "EXP_experiments/TLAST_improved/PEMS07/TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint",
        ],
        "Lite-R256": [
            f"EXP_experiments/TLAST_improved/PEMS07/TLAST_improved_PEMS07_12to12_exppems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}"
            for seed in (1, 2, 3)
        ],
    },
}

ANCHOR_RE = re.compile(r"Epoch 000 anchor,.*?Valid MAE: ([0-9.]+).*?Valid RMSE: ([0-9.]+)")
EPOCH_RE = re.compile(
    r"Epoch: (\d+), Train Loss: ([0-9.]+), Train MAE: ([0-9.]+).*?"
    r"Valid Loss: ([0-9.]+), Valid MAE: ([0-9.]+).*?Valid RMSE: ([0-9.]+), "
    r"Training Time: ([0-9.]+)/epoch"
)


def parse_run(directory: Path):
    logs = list(directory.glob("*.log"))
    if len(logs) != 1:
        raise RuntimeError(f"expected one log under {directory}, found {logs}")
    text = logs[0].read_text(encoding="utf-8", errors="ignore")
    anchor = ANCHOR_RE.search(text)
    rows = []
    if anchor:
        rows.append({
            "epoch": 0,
            "train_loss": None,
            "train_mae": None,
            "val_loss": float(anchor.group(1)),
            "val_mae": float(anchor.group(1)),
            "val_rmse": float(anchor.group(2)),
            "training_seconds": 0.0,
        })
    for match in EPOCH_RE.finditer(text):
        rows.append({
            "epoch": int(match.group(1)),
            "train_loss": float(match.group(2)),
            "train_mae": float(match.group(3)),
            "val_loss": float(match.group(4)),
            "val_mae": float(match.group(5)),
            "val_rmse": float(match.group(6)),
            "training_seconds": float(match.group(7)),
        })
    if not rows:
        raise RuntimeError(f"no epoch rows in {logs[0]}")
    best = min(rows, key=lambda row: row["val_mae"])
    times = [row["training_seconds"] for row in rows if row["epoch"] > 0]
    return rows, {
        "log": str(logs[0].resolve()),
        "epochs_completed": int(max(row["epoch"] for row in rows)),
        "best_epoch": int(best["epoch"]),
        "best_val_mae": float(best["val_mae"]),
        "mean_training_seconds_per_epoch": float(np.mean(times)),
        "total_training_seconds_from_epoch_logs": float(np.sum(times)),
        "training_finished_marker": "Training finished" in text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = {"protocol": "immutable_log_convergence_v1", "datasets": {}}
    plot_payload = {}

    for dataset, methods in RUNS.items():
        summary["datasets"][dataset] = {}
        plot_payload[dataset] = {}
        for method, directories in methods.items():
            parsed = []
            run_summaries = []
            for seed, relative in enumerate(directories, start=1):
                rows, run_summary = parse_run(ROOT / relative)
                parsed.append(rows)
                run_summary["seed"] = seed
                run_summaries.append(run_summary)
                for row in rows:
                    all_rows.append({"dataset": dataset, "method": method, "seed": seed, **row})
            common_epochs = sorted(set.intersection(*[
                {row["epoch"] for row in rows} for rows in parsed
            ]))
            means, stds = [], []
            for epoch in common_epochs:
                values = np.asarray([
                    next(row["val_mae"] for row in rows if row["epoch"] == epoch)
                    for rows in parsed
                ], dtype=np.float64)
                means.append(float(values.mean()))
                stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
            plot_payload[dataset][method] = (np.asarray(common_epochs), np.asarray(means), np.asarray(stds))
            summary["datasets"][dataset][method] = {
                "runs": run_summaries,
                "common_epoch_count": len(common_epochs),
                "final_common_epoch": int(common_epochs[-1]),
                "final_common_val_mae_mean": means[-1],
                "final_common_val_mae_sample_std": stds[-1],
            }

    with (out / "convergence_epochs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Matplotlib is not guaranteed in the frozen experiment runtime.  Render
    # with Pillow so curve generation remains a zero-install, zero-training
    # post-processing step.
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1700, 680
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)
    colors = {"Teacher": "#C44E52", "Lite-R256": "#4C72B0"}
    margins = {"left": 140, "right": 45, "top": 70, "bottom": 90}
    panel_gap = 90
    panel_width = (width - margins["left"] - margins["right"] - panel_gap) // 2
    panel_height = height - margins["top"] - margins["bottom"]

    for panel, dataset in enumerate(("PEMS04", "PEMS07")):
        x0 = margins["left"] + panel * (panel_width + panel_gap)
        y0 = margins["top"]
        payload = plot_payload[dataset]
        x_max = max(int(values[0].max()) for values in payload.values())
        all_y = np.concatenate([
            np.r_[values[1] - values[2], values[1] + values[2]]
            for values in payload.values()
        ])
        y_pad = max(float(np.ptp(all_y)) * 0.08, 0.002)
        y_min, y_max = float(all_y.min() - y_pad), float(all_y.max() + y_pad)

        def xy(epoch, value):
            x = x0 + panel_width * float(epoch) / max(x_max, 1)
            y = y0 + panel_height * (y_max - float(value)) / (y_max - y_min)
            return x, y

        for tick in np.linspace(y_min, y_max, 6):
            _, py = xy(0, tick)
            draw.line((x0, py, x0 + panel_width, py), fill="#DDDDDD", width=1)
            draw.text((x0 - 85, py - 10), f"{tick:.3f}", fill="#333333", font=small)
        draw.line((x0, y0, x0, y0 + panel_height), fill="black", width=2)
        draw.line((x0, y0 + panel_height, x0 + panel_width, y0 + panel_height), fill="black", width=2)
        step = 5 if x_max >= 20 else 2
        for epoch in range(0, x_max + 1, step):
            px, py = xy(epoch, y_min)
            draw.line((px, py, px, py + 7), fill="black", width=2)
            draw.text((px - 9, py + 14), str(epoch), fill="#333333", font=small)

        for method, (epochs, mean, std) in payload.items():
            color = colors[method]
            if np.any(std > 0):
                upper = [xy(e, m + s) for e, m, s in zip(epochs, mean, std)]
                lower = [xy(e, m - s) for e, m, s in zip(epochs[::-1], mean[::-1], std[::-1])]
                overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
                ImageDraw.Draw(overlay).polygon(upper + lower, fill=color + "28")
                canvas.paste(overlay, (0, 0), overlay)
                draw = ImageDraw.Draw(canvas)
            points = [xy(e, m) for e, m in zip(epochs, mean)]
            draw.line(points, fill=color, width=4, joint="curve")
            for px, py in points:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)

        draw.text((x0 + panel_width / 2 - 45, 22), dataset, fill="black", font=font)
        draw.text((x0 + panel_width / 2 - 30, height - 45), "Epoch", fill="black", font=font)
        legend_x, legend_y = x0 + panel_width - 190, y0 + 18
        for row, method in enumerate(("Teacher", "Lite-R256")):
            ly = legend_y + row * 30
            draw.line((legend_x, ly + 8, legend_x + 38, ly + 8), fill=colors[method], width=4)
            draw.text((legend_x + 48, ly), method, fill="black", font=small)

    rotated = Image.new("RGBA", (180, 40), (255, 255, 255, 0))
    ImageDraw.Draw(rotated).text((0, 5), "Validation MAE", fill="black", font=font)
    rotated = rotated.rotate(90, expand=True)
    canvas.paste(rotated, (5, height // 2 - rotated.height // 2), rotated)
    canvas.save(out / "validation_mae_convergence.png", dpi=(260, 260))
    canvas.save(out / "validation_mae_convergence.pdf", "PDF", resolution=260.0)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
