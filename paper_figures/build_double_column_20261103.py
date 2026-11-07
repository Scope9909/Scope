"""Render two publication-width, two-panel DARC experiment figures.

All plotted numbers are copied from the checked source tables and the
existing figure-data audit. This script does not run model experiments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "_runtime_deps" / "darc_figures_20261103"
sys.path.insert(0, str(DEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SOURCE = ROOT / "output" / "pdf" / "darc_single_column_20261103" / "figure_data.json"
TEX = ROOT / "Template.tex"
OUT = ROOT / "output" / "pdf" / "darc_double_column_20261103"
OUT.mkdir(parents=True, exist_ok=True)

FONT = Path(r"C:\Windows\Fonts\times.ttf")
assert FONT.exists(), FONT
font_manager.fontManager.addfont(str(FONT))
FONT_NAME = font_manager.FontProperties(fname=str(FONT)).get_name()
BOLD_FONT_FILE = Path(r"C:\Windows\Fonts\timesbd.ttf")
assert BOLD_FONT_FILE.exists(), BOLD_FONT_FILE
font_manager.fontManager.addfont(str(BOLD_FONT_FILE))
BOLD_CAPTION_FONT = font_manager.FontProperties(fname=str(BOLD_FONT_FILE), size=9.4)

plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "font.size": 8.0,
        "font.weight": "normal",
        "text.color": "black",
        "axes.labelsize": 8.2,
        "axes.labelcolor": "black",
        "xtick.labelsize": 7.4,
        "xtick.color": "black",
        "ytick.labelsize": 7.4,
        "ytick.color": "black",
        "legend.fontsize": 7.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

NAVY = "#183450"
BLUE = "#246795"
SKY = "#70A8CB"
TEAL = "#148C7B"
ORANGE = "#D57429"
LAVENDER = "#785AA7"
GRID = "#DFEAF1"
INK = "#000000"


def style_axis(ax, direction="y"):
    ax.grid(axis=direction, color=GRID, linewidth=0.65, zorder=0)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="both", colors=INK, length=3.2, width=0.7)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    arrow = dict(arrowstyle="-|>", color=INK, lw=0.9,
                 mutation_scale=8, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(1.0, 0), xytext=(0.955, 0),
                xycoords="axes fraction", arrowprops=arrow,
                annotation_clip=False, zorder=10)
    ax.annotate("", xy=(0, 1.0), xytext=(0, 0.945),
                xycoords="axes fraction", arrowprops=arrow,
                annotation_clip=False, zorder=10)


def save_all(fig, stem):
    pdf = OUT / f"{stem}.pdf"
    svg = OUT / f"{stem}.svg"
    png = OUT / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(svg, dpi=300)
    fig.savefig(png, dpi=360)
    plt.close(fig)
    return [str(pdf), str(svg), str(png)]


def storage_figure(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.95))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.77, wspace=0.30)

    names = ["Teacher base", "Student base", "Full-node exact", "SRM exact", "DARC"]
    colors = [SKY, BLUE, NAVY, TEAL, ORANGE]
    markers = ["^", "o", "s", "D", "o"]
    sizes = [62, 54, 59, 71, 47]
    stage = data["stages"]
    assert len(stage) == 5
    for i, (name, color, marker, size) in enumerate(zip(names, colors, markers, sizes)):
        ax1.scatter(
            stage[i]["total_bytes"] / 2**20,
            stage[i]["calibrated_mae"],
            s=size,
            marker=marker,
            c=color,
            edgecolors="white",
            linewidths=0.55,
            zorder=4 + (i == 4),
            label=name,
        )
    ax1.set_xlim(3.7, 16.0)
    ax1.set_ylim(18.47, 18.82)
    ax1.set_xticks([4, 8, 12, 16])
    ax1.set_yticks([18.5, 18.6, 18.7, 18.8])
    ax1.set_xlabel("Total stored assets (MiB)")
    ax1.set_ylabel("Calibrated test MAE")
    style_axis(ax1)
    ax1.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.025), ncol=2,
        frameon=False, columnspacing=0.80, handletextpad=0.30,
        borderaxespad=0.0, labelspacing=0.25, markerscale=0.72,
    )

    percentages = np.asarray(data["codec_percent_of_npz"], dtype=float)
    assert percentages.shape == (4, 4)
    assert np.allclose(percentages[:, 0], 100.0)
    assert percentages[:, 1:].max() < 75
    y = np.arange(4)
    offsets = [-0.22, 0.0, 0.22]
    bar_colors = [SKY, BLUE, ORANGE]
    bar_names = ["Direct", "Fully exact", "Final"]
    for j in range(3):
        ax2.barh(y + offsets[j], percentages[:, j + 1], height=0.205,
                 color=bar_colors[j], edgecolor="none", zorder=3, label=bar_names[j])
    ax2.set_yticks(y, data["datasets"])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 75)
    ax2.set_xticks([0, 15, 30, 45, 60, 75])
    ax2.set_xlabel("File size (% of indexed NPZ)")
    style_axis(ax2, "x")
    ax2.grid(False)
    ax2.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.025), ncol=3,
        frameon=False, columnspacing=0.6, handletextpad=0.35,
        borderaxespad=0.0,
    )

    for ax, label in zip(
        (ax1, ax2),
        ("(a) Accuracy vs. stored assets", "(b) Retrieval-state file size"),
    ):
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, 0.06, label, ha="center",
                 va="bottom", color=INK, fontproperties=BOLD_CAPTION_FONT)
    return save_all(fig, "fig_darc_accuracy_storage_and_codec_v4")


def lowrank_figure():
    tex = TEX.read_text(encoding="utf-8")
    for evidence in ["18.4407 & 19.2691 & 21.0249", "17.5485 & 17.7523 & 19.1405", "17.8318 \\pm 0.0014", "17.8325 \\pm 0.0008"]:
        assert evidence in tex, f"Source table changed: {evidence}"

    parent = np.array([17.5297, 17.7279, 19.1275])
    rank128 = {
        "Weight SVD": (np.array([18.4407, 19.2691, 21.0249]), BLUE),
        "Diagonal moment": (np.array([18.0642, 18.5744, 20.1656]), LAVENDER),
        "Full moment": (np.array([17.5485, 17.7523, 19.1405]), TEAL),
        "Full + fine-tune": (np.array([17.5397, 17.7280, 19.1361]), ORANGE),
    }
    # The rank-256 validation comparison is panel (a); the separate
    # rank-128 Train-OOF fidelity comparison is panel (b).
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(7.16, 3.22))
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.23, top=0.83, wspace=0.45)

    y = np.arange(4)
    for row, (name, (mae, color)) in enumerate(rank128.items()):
        gap = mae - parent
        ax1.barh(row, gap.mean(), height=0.50, color=color, alpha=0.76, zorder=2)
        ax1.scatter(gap, row + np.array([-0.11, 0.0, 0.11]), s=28, c=color,
                    edgecolors=INK, linewidths=0.25, zorder=4)
    ax1.axvline(0, color=INK, linewidth=0.75, zorder=1)
    ax1.set_yticks(y, list(rank128))
    ax1.invert_yaxis()
    ax1.set_xlim(-0.09, 2.05)
    ax1.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax1.set_xlabel("MAE increase vs. Parent")
    style_axis(ax1, "x")
    ax1.legend(
        [Patch(facecolor=BLUE, alpha=0.76), Line2D([0], [0], marker="o", color=BLUE,
                                                 linestyle="none", markerfacecolor=BLUE,
                                                 markeredgecolor=INK, markersize=5)],
        ["Mean gap", "Three folds"],
        loc="lower left", bbox_to_anchor=(0.0, 1.025), ncol=2,
        frameon=False, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0,
    )

    # The full-moment gaps cluster near zero on the common scale. The inset
    # reuses the same observations, with its own explicitly labeled scale.
    zoom1 = ax1.inset_axes([0.54, 0.18, 0.43, 0.30], zorder=6)
    for row, name in enumerate(["Full moment", "Full + fine-tune"]):
        mae, color = rank128[name]
        gap = mae - parent
        zoom1.barh(row, gap.mean(), height=0.37, color=color, alpha=0.76, zorder=2)
        zoom1.scatter(gap, row + np.array([-0.08, 0.0, 0.08]), s=12,
                      c=color, edgecolors=INK, linewidths=0.18, zorder=4)
    zoom1.set_xlim(0, 0.03)
    zoom1.set_xticks([0, 0.015, 0.03], ["0", ".015", ".030"])
    zoom1.set_yticks([0, 1], ["No FT", "+ FT"])
    zoom1.invert_yaxis()
    zoom1.set_title("Full-moment detail", fontsize=6.8, pad=1.5)
    zoom1.tick_params(axis="both", labelsize=6.1, length=2, width=0.45, pad=1)
    zoom1.set_facecolor("white")
    for spine in zoom1.spines.values():
        spine.set_visible(True)
        spine.set_color(NAVY)
        spine.set_linewidth(0.5)

    rank256 = [
        ("SVD, no KD", 18.2097, 17.8647, BLUE),
        ("SVD + KD", 18.2097, 17.8318, BLUE),
        ("Moment, no KD", 17.8542, 17.8435, TEAL),
        ("Moment + KD", 17.8542, 17.8325, TEAL),
    ]
    for row, (_, initial, best, color) in enumerate(rank256):
        ax2.plot([best, initial], [row, row], color=color, lw=1.45, zorder=2)
        ax2.scatter(initial, row, s=35, c=color, marker="s",
                    edgecolors="white", linewidths=0.38, zorder=3)
        ax2.scatter(best, row, s=45, c=color, marker="D",
                    edgecolors="white", linewidths=0.38, zorder=4)
    ax2.set_yticks(np.arange(4), [item[0] for item in rank256])
    ax2.invert_yaxis()
    ax2.set_xlim(17.79, 18.235)
    ax2.set_xticks([17.8, 17.9, 18.0, 18.1, 18.2])
    ax2.set_xlabel("Validation MAE (lower is better)")
    style_axis(ax2, "x")
    ax2.legend(
        [Line2D([0], [0], marker="s", color=NAVY, linestyle="none", markerfacecolor=NAVY,
                markeredgecolor=NAVY, markersize=5),
         Line2D([0], [0], marker="D", color=NAVY, linestyle="none", markerfacecolor=NAVY,
                markeredgecolor=NAVY, markersize=5)],
        ["Epoch 0", "Best validation"],
        loc="lower left", bbox_to_anchor=(0.0, 1.025), ncol=2,
        frameon=False, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.0,
    )

    # A second zoom makes the two full-moment endpoints legible without
    # distorting the main dumbbell axis used for the SVD comparison.
    zoom2 = ax2.inset_axes([0.55, 0.18, 0.42, 0.30], zorder=6)
    for row, (_, initial, best, color) in enumerate(rank256[2:]):
        zoom2.plot([best, initial], [row, row], color=color, lw=1.25, zorder=2)
        zoom2.scatter(initial, row, s=18, c=color, marker="s",
                      edgecolors="white", linewidths=0.2, zorder=3)
        zoom2.scatter(best, row, s=23, c=color, marker="D",
                      edgecolors="white", linewidths=0.2, zorder=4)
    zoom2.set_xlim(17.825, 17.862)
    zoom2.set_xticks([17.83, 17.85])
    zoom2.set_yticks([0, 1], ["No KD", "+ KD"])
    zoom2.invert_yaxis()
    zoom2.set_title("Full-moment detail", fontsize=6.8, pad=1.5)
    zoom2.tick_params(axis="both", labelsize=6.1, length=2, width=0.45, pad=1)
    zoom2.set_facecolor("white")
    for spine in zoom2.spines.values():
        spine.set_visible(True)
        spine.set_color(NAVY)
        spine.set_linewidth(0.5)

    for ax, label in zip(
        (ax2, ax1),
        ("(a) Rank-256 initialization and KD", "(b) Rank-128 initialization"),
    ):
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, 0.06, label, ha="center",
                 va="bottom", color=INK, fontproperties=BOLD_CAPTION_FONT)
    return save_all(fig, "fig_darc_iasc_initialization_and_kd_v5")


if __name__ == "__main__":
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    outputs = {"storage": storage_figure(data), "lowrank": lowrank_figure()}
    (OUT / "figure_sources.json").write_text(
        json.dumps({"source_files": [str(SOURCE), str(TEX)], "outputs": outputs}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(outputs, indent=2))
