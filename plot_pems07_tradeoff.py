"""PEMS07 storage-accuracy trade-off from Template (13).tex, Table VI.

The figure emphasizes the joint change in total stored assets and calibrated
test MAE.  The two SRM configurations intentionally overlap at this scale;
their exact values remain in the table.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


OUT = Path(__file__).resolve().parent / "output" / "pdf" / "performance_redrawn"
OUT.mkdir(parents=True, exist_ok=True)

for font_file in ("C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"):
    font_manager.fontManager.addfont(font_file)

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "stix",
        "savefig.facecolor": "white",
    }
)

DATA = {
    "Teacher base": (6406.00, 18.7532),
    "Student base": (5008.76, 18.7793),
    "Full-node exact": (14960.79, 18.7459),
    "SRM exact": (5015.39, 18.5091),
    "DARC (SRM + Q8)": (5015.38, 18.5115),
}

COLORS = {
    "reference": "#8A969F",
    "student": "#7594AD",
    "full": "#BD9477",
    "srm": "#5B8B73",
    "darc": "#1F5A85",
    "ink": "#303C46",
    "grid": "#DDE3E7",
}

fig, ax = plt.subplots(figsize=(9.5, 4.6))
fig.subplots_adjust(left=0.115, right=0.985, top=0.92, bottom=0.20)

ax.set_xlim(4.25, 15.35)
ax.set_ylim(18.46, 18.82)
ax.set_xticks([5, 7.5, 10, 12.5, 15])
ax.set_yticks([18.5, 18.6, 18.7, 18.8])
ax.set_axisbelow(True)
ax.grid(axis="both", color=COLORS["grid"], linewidth=0.7, linestyle=(0, (2.5, 3.5)))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(COLORS["ink"])

ax.set_xlabel("Total stored assets (MiB)", color=COLORS["ink"], labelpad=12)
ax.set_ylabel("Calibrated test MAE", color=COLORS["ink"], labelpad=12)
ax.tick_params(axis="both", colors=COLORS["ink"], labelsize=12, length=4)


def xy(name):
    total_kib, mae = DATA[name]
    return total_kib / 1024.0, mae


# Show the substantive transition under the same frozen Student.
full_x, full_y = xy("Full-node exact")
srm_x, srm_y = xy("SRM exact")
ax.annotate(
    "",
    xy=(srm_x + 0.10, srm_y + 0.007),
    xytext=(full_x - 0.12, full_y - 0.006),
    arrowprops={"arrowstyle": "-|>", "color": COLORS["srm"], "lw": 2.0, "mutation_scale": 15},
    zorder=2,
)

styles = [
    ("Teacher base", "reference", "o", 105),
    ("Student base", "student", "s", 112),
    ("Full-node exact", "full", "D", 125),
    ("SRM exact", "srm", "o", 215),
    ("DARC (SRM + Q8)", "darc", "D", 85),
]
for name, color_key, marker, area in styles:
    x, y = xy(name)
    ax.scatter(
        x, y, s=area, marker=marker, color=COLORS[color_key],
        edgecolor="white", linewidth=1.25, zorder=4,
    )

# Direct labels keep the graphic self-contained and avoid a large legend.
ax.annotate("Student base", xy("Student base"), xytext=(3, 11),
            textcoords="offset points", ha="left", color=COLORS["student"],
            fontsize=13, fontweight="bold")
ax.annotate("Teacher base", xy("Teacher base"), xytext=(8, 11),
            textcoords="offset points", ha="left", color=COLORS["reference"],
            fontsize=13, fontweight="bold")
ax.annotate("Full-node exact", xy("Full-node exact"), xytext=(-4, 12),
            textcoords="offset points", ha="right", color=COLORS["full"],
            fontsize=13, fontweight="bold")

# SRM exact and Q8 differ by only 0.01 KiB in total storage and 0.0024 MAE.
# At the full-range scale they correctly appear as an overlapping pair.
ax.annotate("SRM exact / DARC (Q8)", xy("SRM exact"), xytext=(15, -2),
            textcoords="offset points", ha="left", va="center",
            color=COLORS["darc"], fontsize=13, fontweight="bold")

reduction_pct = 100 * (1 - DATA["SRM exact"][0] / DATA["Full-node exact"][0])
mae_gain = DATA["Full-node exact"][1] - DATA["SRM exact"][1]
ax.text(
    9.65, 18.635,
    f"SRM: {reduction_pct:.1f}% less total storage\n"
    f"and {mae_gain:.4f} lower MAE",
    ha="center", va="center", fontsize=12.5, color=COLORS["srm"],
    fontweight="bold",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2},
)

base = OUT / "pems07-storage-accuracy-tradeoff"
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.12)
fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
print(base)
