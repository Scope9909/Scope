"""Two editable, table-backed experimental figures for the DARC paper.

Figure A uses Table 3's *early-format*, later-Validation-fold SRM diagnostic.
Figure B uses Table 7(a)'s frozen Test retrieval-state file bytes.
The two experiments have different scopes and are deliberately not combined.
"""

from pathlib import Path

from build_pems07_visual_evidence_svg import E, L, R, S, T


ROOT = Path(__file__).resolve().parent
INK = "#25354a"
MUTED = "#65758a"
GRID = "#dfe5eb"
LIGHT = "#e9edf1"
BLUE = "#477fb2"
GREEN = "#137e67"
ORANGE = "#c77c3f"

# Paper Table 3: node count, full/sparse state MiB, and raw ΔMAE for each of
# three later Validation folds (sparse minus full). No Test values appear here.
SRM = [
    ("PEMS03", 36, 358, 8.472, 0.582, (0.010, 0.002, -0.020)),
    ("PEMS04", 31, 307, 4.779, 0.209, (0.004, 0.023, 0.004)),
    ("PEMS07", 121, 883, 23.503, 1.965, (-0.040, -0.235, -0.274)),
    ("PEMS08", 20, 170, 2.778, 0.129, (0.145, 0.119, 0.143)),
]

# Paper Table 7(a): frozen Test files, bytes in dataset order 03/04/07/08.
CODEC = {
    "Original NPZ": (275236, 78718, 355339, 32901),
    "Same-container generic": (190096, 46263, 95082, 18058),
    "All-exact adaptive": (166400, 38511, 83063, 14984),
    "Final state": (166067, 38511, 80752, 14984),
}


def circle(x, y, r, fill):
    return E("circle", {"cx": round(x, 2), "cy": round(y, 2), "r": r, "fill": fill})


def srm_tradeoff():
    width, height = 1100, 485
    a = [R(0, 0, width, height, "white")]
    a += [T(250, 43, "(a) Retrieval state retained", 20, True),
          T(690, 43, "(b) Prediction change after SRM", 20, True),
          T(30, 86, "Dataset", 15, color=MUTED),
          T(160, 86, "Nodes", 15, color=MUTED),
          T(258, 86, "Full state = 100%", 15, color=MUTED),
          T(697, 86, "Three subsequent Validation folds", 15, color=MUTED)]

    rows_y = (143, 219, 295, 371)
    bar_x, bar_w = 253, 325
    plot_x0, plot_x1 = 690, 1052
    xmin, xmax = -0.30, 0.18

    def px(v):
        return plot_x0 + (v - xmin) / (xmax - xmin) * (plot_x1 - plot_x0)

    for tick in (-0.3, -0.2, -0.1, 0.0, 0.1):
        x = px(tick)
        a += [L(x, 103, x, 399, INK if tick == 0 else GRID, 1.5 if tick == 0 else 1),
              T(x, 425, f"{tick:+.1f}" if tick else "0", 14, color=MUTED, anchor="middle")]
    a += [T(plot_x0 + 93, 465, "SRM lower MAE", 16, color=GREEN, anchor="middle"),
          T(plot_x1 - 48, 465, "Full lower MAE", 16, color=ORANGE, anchor="middle"),
          T((plot_x0 + plot_x1) / 2, 446, "Raw ΔMAE (SRM − full)", 17, anchor="middle")]

    for (name, kept, all_nodes, full_mib, sparse_mib, deltas), y in zip(SRM, rows_y):
        retained = sparse_mib / full_mib * 100
        assert 0 < retained < 10
        a += [L(30, y + 36, 1062, y + 36, GRID),
              T(30, y + 6, name, 19, True),
              T(159, y + 6, f"{kept}/{all_nodes}", 16, color=MUTED),
              R(bar_x, y - 13, bar_w, 25, LIGHT),
              R(bar_x, y - 13, retained / 100 * bar_w, 25, GREEN),
              T(bar_x + 39, y + 6, f"{retained:.1f}% retained", 16, True)]
        for j, v in enumerate(deltas):
            a.append(circle(px(v), y + (j - 1) * 9, 5.2, GREEN if v < 0 else ORANGE))

    a += [T(30, 465, "Early state format · later Validation folds only", 14, color=MUTED)]
    svg = S(width, height, "SRM precision-storage tradeoff on four PEMS datasets",
            "Left: early-format sparse retrieval states retain 4.4 to 8.4 percent of the full-node state. Right: three later Validation folds per dataset show raw MAE change, sparse minus full. Negative favors SRM; PEMS08 has positive changes.", a)
    (ROOT / "fig_darc_srm_precision_storage.svg").write_text(svg, encoding="utf-8")


def codec_storage():
    width, height = 1100, 485
    totals = {name: sum(vals) for name, vals in CODEC.items()}
    assert totals == {"Original NPZ": 742194, "Same-container generic": 349499,
                      "All-exact adaptive": 302958, "Final state": 300314}
    base = totals["Original NPZ"]
    x0, x1 = 300, 910
    scale = (x1 - x0) / 800000
    colors = {"Original NPZ": LIGHT, "Same-container generic": BLUE,
              "All-exact adaptive": "#6ca9a1", "Final state": GREEN}
    ys = (114, 201, 288, 375)
    a = [R(0, 0, width, height, "white"),
         T(34, 42, "Four-dataset Test retrieval-state files", 22, True),
         T(34, 70, "Only file bytes are compared; network and calibrator are excluded", 15, color=MUTED),
         T(1065, 42, "−59.54% vs. NPZ", 20, True, color=GREEN, anchor="end")]

    for tick in (0, 200, 400, 600, 800):
        x = x0 + tick * 1000 * scale
        a += [L(x, 90, x, 420, GRID),
              T(x, 445, str(tick), 15, color=MUTED, anchor="middle")]
    a += [T((x0 + x1) / 2, 470, "Four-file total (thousand bytes)", 17, anchor="middle")]

    for (name, value), y in zip(totals.items(), ys):
        w = value * scale
        a += [T(279, y + 6, name, 17, True, anchor="end"),
              R(x0, y - 14, w, 29, colors[name]),
              T(x0 + w + 12, y + 7, f"{value:,} B", 17, True)]

    a += [T(34, 427, "Final vs. all-exact: 2,644 B smaller", 15, color=MUTED)]
    svg = S(width, height, "DARC retrieval-state encoding storage comparison",
            "Four-dataset frozen Test retrieval-state files: original NPZ 742194 bytes, same-container generic compression 349499, all-exact adaptive coding 302958, final state 300314. Final storage is 59.54 percent less than NPZ; the Q8 increment beyond all-exact is 2644 bytes.", a)
    (ROOT / "fig_darc_codec_storage.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    srm_tradeoff()
    codec_storage()
    print(ROOT / "fig_darc_srm_precision_storage.svg")
    print(ROOT / "fig_darc_codec_storage.svg")
