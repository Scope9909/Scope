"""Editable accuracy--storage scatter for the matched PEMS07 seed-1 run."""

from pathlib import Path

from build_pems07_visual_evidence_svg import E, L, R, S, T


OUT = Path(__file__).resolve().parent
INK = "#25354a"
MUTED = "#647487"
GRID = "#e2e8ed"
BLUE = "#3975ad"
AMBER = "#c27a33"
GREEN = "#117b64"

# Checkpoint + retrieval state + configuration-specific calibrator, in bytes.
# All four rows use the same frozen Student, PEMS07 seed 1 and validation-fitted
# calibration. The unmatched Teacher-base reference is intentionally omitted.
ROWS = [
    ("Student base", 5_128_969, 18.7793, BLUE),
    ("Full-node exact", 15_319_851, 18.7459, AMBER),
    ("SRM exact", 5_135_762, 18.5091, GREEN),
    ("DARC (SRM + Q8)", 5_135_748, 18.5115, GREEN),
]


def build():
    width, height = 1050, 500
    left, right, top, bottom = 132, 985, 66, 392
    xmin, xmax, ymin, ymax = 4.0, 16.0, 18.45, 18.85

    def xy(size_bytes, mae):
        size_mib = size_bytes / (1024 * 1024)
        return (left + (size_mib - xmin) / (xmax - xmin) * (right - left),
                bottom - (mae - ymin) / (ymax - ymin) * (bottom - top))

    a = [R(0, 0, width, height, "white")]
    for tick in (4, 6, 8, 10, 12, 14, 16):
        x = left + (tick - xmin) / (xmax - xmin) * (right - left)
        a += [L(x, top, x, bottom, GRID), T(x, bottom + 27, str(tick), 15, color=MUTED, anchor="middle")]
    for tick in (18.5, 18.6, 18.7, 18.8):
        y = bottom - (tick - ymin) / (ymax - ymin) * (bottom - top)
        a += [L(left, y, right, y, GRID), T(left - 14, y + 5, f"{tick:.1f}", 15, color=MUTED, anchor="end")]
    a += [L(left, top, left, bottom, INK, 1.4), L(left, bottom, right, bottom, INK, 1.4)]
    a += [T((left + right) / 2, 455, "Checkpoint + retrieval state + calibrator (MiB)", 18, anchor="middle"),
          T(41, (top + bottom) / 2, "Calibrated test MAE ↓", 18, anchor="middle", rotate=True),
          T(left, 35, "PEMS07 · matched seed-1 configurations", 21, True),
          T(right, 35, "Lower left is better", 15, color=MUTED, anchor="end")]

    # Render the close pair in true coordinates, without jitter. The larger
    # diamond remains visible behind the smaller exact-state circle.
    x_base, y_base = xy(ROWS[0][1], ROWS[0][2])
    x_full, y_full = xy(ROWS[1][1], ROWS[1][2])
    x_srm, y_srm = xy(ROWS[2][1], ROWS[2][2])
    x_darc, y_darc = xy(ROWS[3][1], ROWS[3][2])
    assert abs(x_srm - x_darc) < 0.002 and abs(y_srm - y_darc) < 2.1

    a += [E("circle", {"cx": x_base, "cy": y_base, "r": 8, "fill": BLUE}),
          T(x_base + 17, y_base - 10, "Student base", 17, True),
          T(x_base + 17, y_base + 12, "18.7793 MAE", 15, color=MUTED),
          E("circle", {"cx": x_full, "cy": y_full, "r": 8, "fill": AMBER}),
          T(x_full - 18, y_full - 11, "Full-node exact", 17, True, anchor="end"),
          T(x_full - 18, y_full + 11, "18.7459 MAE", 15, color=MUTED, anchor="end")]

    # DARC diamond plus the exact-state circle share practically the same
    # coordinates. The exact difference is stated under the direct label.
    d = 13
    a += [E("polygon", {"points": f"{x_darc},{y_darc-d} {x_darc+d},{y_darc} {x_darc},{y_darc+d} {x_darc-d},{y_darc}",
                        "fill": "white", "stroke": GREEN, "stroke_width": 2.4}),
          E("circle", {"cx": x_srm, "cy": y_srm, "r": 5.5, "fill": GREEN}),
          L(x_srm + 14, y_srm, x_srm + 32, y_srm, GREEN, 1.4),
          T(x_srm + 40, y_srm - 4, "SRM exact ≈ DARC", 17, True),
          T(x_srm + 40, y_srm + 18, "18.5091 vs. 18.5115 MAE", 15, color=MUTED)]

    # The note reports why the pair is visually indistinguishable. Its storage
    # values are full three-asset totals, not retrieval-state file sizes.
    a += [L(left, 472, right, 472, GRID),
          T(left, 493, "SRM exact → DARC: total −14 B; MAE +0.0024. The two marks nearly overlap at this scale.", 14, color=MUTED)]

    svg = S(width, height, "PEMS07 accuracy versus stored assets",
            "Matched seed-1 configurations. Horizontal axis is checkpoint plus retrieval-state file plus calibrator in MiB; vertical axis is calibrated test MAE. SRM exact and DARC nearly coincide, differing by only 14 bytes and 0.0024 MAE.", a)
    (OUT / "fig_pems07_accuracy_storage.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    build()
    print(OUT / "fig_pems07_accuracy_storage.svg")
