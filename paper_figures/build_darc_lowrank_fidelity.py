"""Editable PEMS04 low-rank fidelity figure from paper Table 4."""

from pathlib import Path

from build_pems07_visual_evidence_svg import E, L, R, S, T


ROOT = Path(__file__).resolve().parent
INK, MUTED, GRID = "#25354a", "#65758a", "#dfe5eb"
BLUE, ORANGE, GREEN, DARK_GREEN = "#477fb2", "#c77c3f", "#72aaa0", "#137e67"

# PEMS04 rank-128 Train-OOF MAE, three matched folds (paper Table 4).
PARENT = (17.5297, 17.7279, 19.1275)
ROWS = (
    ("Weight SVD", (18.4407, 19.2691, 21.0249), BLUE),
    ("Diagonal moment", (18.0642, 18.5744, 20.1656), ORANGE),
    ("Full second moment", (17.5485, 17.7523, 19.1405), GREEN),
    ("Full moment + fine-tune", (17.5397, 17.7280, 19.1361), DARK_GREEN),
)


def build():
    width, height = 1050, 450
    x0, x1 = 312, 900
    vmax = 2.0
    ys = (116, 191, 266, 341)
    a = [R(0, 0, width, height, "white"),
         T(36, 41, "Low-rank prediction fidelity · PEMS04", 22, True),
         T(36, 68, "MAE increase over the matched Parent on each Train-OOF fold", 16, color=MUTED),
         T(1030, 68, "Lower is better", 15, color=MUTED, anchor="end"),
         T(1030, 88, "Mean gap", 15, color=MUTED, anchor="end")]

    for tick in (0, 0.5, 1.0, 1.5, 2.0):
        x = x0 + tick / vmax * (x1 - x0)
        a += [L(x, 86, x, 370, INK if tick == 0 else GRID, 1.3 if tick == 0 else 1),
              T(x, 397, f"{tick:g}", 15, color=MUTED, anchor="middle")]
    a += [T((x0 + x1) / 2, 430, "ΔMAE relative to Parent (mean bar; three fold dots)", 17, anchor="middle")]

    for (name, mae, color), y in zip(ROWS, ys):
        gaps = tuple(v - p for v, p in zip(mae, PARENT))
        assert all(g >= 0 for g in gaps)
        mean = sum(gaps) / 3
        bar_end = x0 + mean / vmax * (x1 - x0)
        a += [T(x0 - 19, y + 6, name, 18, True, anchor="end"),
              R(x0, y - 9, max(1.5, bar_end - x0), 18, color)]
        for j, gap in enumerate(gaps):
            x = x0 + gap / vmax * (x1 - x0)
            a.append(E("circle", {"cx": round(x, 2), "cy": y + (j - 1) * 8,
                                   "r": 4.3, "fill": "white", "stroke": color,
                                   "stroke_width": 2.2}))
        a.append(T(1030, y + 6, f"{mean:.3f}", 16, True, color=INK, anchor="end"))

    svg = S(width, height, "PEMS04 low-rank initialization fidelity",
            "Rank-128 Train-OOF three-fold MAE increase over the same-fold Parent. Weight SVD and diagonal-moment initialization have much larger errors than the full-second-moment initialization before fine-tuning. Bars show the three-fold mean; open dots show individual folds. These are not final Test results.", a)
    (ROOT / "fig_darc_lowrank_fidelity.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    build()
    print(ROOT / "fig_darc_lowrank_fidelity.svg")
