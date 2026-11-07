"""Generate editable SVG drafts of two ICDE ablation figures.

Values are transcribed from the current Template.tex ablation tables.
No smoothing, interpolation, or statistical aggregation is applied.
"""

from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent
NAVY = "#17365d"
BLUE = "#2c6fb7"
TEAL = "#0d806c"
ORANGE = "#c56b20"
PURPLE = "#7653a6"
RED = "#b44b48"
GRAY = "#626b75"
LIGHT = "#dce3ea"


def svg_open(width, height, title, description):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<style>text{font-family:"Times New Roman",Times,serif;fill:#17365d}'
        '.heading{font-size:27px;font-weight:bold}.label{font-size:23px}'
        '.tick{font-size:20px}.note{font-size:21px}.legend{font-size:20px}</style>',
    ]


def txt(x, y, value, cls="label", anchor=None, fill=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    c = f' fill="{fill}"' if fill else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}"{a}{c}>{escape(str(value))}</text>'


def line(x1, y1, x2, y2, color=LIGHT, width=1.3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{d}/>')


def circle(x, y, radius, color):
    return (f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" '
            f'fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')


def square(x, y, half, color):
    return (f'<rect x="{x-half:.2f}" y="{y-half:.2f}" width="{2*half}" '
            f'height="{2*half}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')


def triangle(x, y, half, color):
    pts = f'{x:.2f},{y-half:.2f} {x-half:.2f},{y+half:.2f} {x+half:.2f},{y+half:.2f}'
    return f'<polygon points="{pts}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'


def write_svg(name, parts):
    path = OUT / name
    path.write_text("\n".join(parts + ["</svg>", ""]), encoding="utf-8")
    print(path)


def srm_figure():
    datasets = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
    folds = [
        [0.010, 0.002, -0.020],
        [0.004, 0.023, 0.004],
        [-0.040, -0.235, -0.274],
        [0.145, 0.119, 0.143],
    ]
    sparse = [0.582, 0.209, 1.965, 0.129]
    full = [8.472, 4.779, 23.503, 2.778]
    retained = [100 * a / b for a, b in zip(sparse, full)]
    p = svg_open(
        780, 370, "SRM accuracy-storage trade-off",
        "Three subsequent validation-fold MAE changes for four PEMS datasets, "
        "and retrieval-state size retained by SRM relative to full-node retrieval.",
    )
    p += [txt(20, 35, "(a) Prediction impact", "heading"),
          txt(475, 35, "(b) State retained", "heading"),
          line(455, 55, 455, 316, LIGHT, 1.5)]

    x0, x1 = 120, 425
    xx = lambda value: x0 + (value + 0.30) / 0.50 * (x1 - x0)
    rows = [105, 155, 205, 255]
    for tick in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]:
        x = xx(tick)
        p.append(line(x, 65, x, 282, GRAY if tick == 0 else LIGHT,
                      1.7 if tick == 0 else 1.0,
                      None if tick == 0 else "3 5"))
        p.append(txt(x, 312, f"{tick:+.1f}" if tick != 0 else "0", "tick", "middle"))
    for i, dataset in enumerate(datasets):
        y = rows[i]
        p.append(txt(18, y + 7, dataset, "label"))
        p.append(line(x0, y, x1, y, LIGHT, 0.8))
        for j, value in enumerate(folds[i]):
            p.append(circle(xx(value), y + [-9, 0, 9][j], 6.2,
                            TEAL if value < 0 else RED))
    p.append(txt(273, 348, "Raw ΔMAE (SRM − full retrieval)", "note", "middle"))

    bx0, bx1 = 545, 695
    for tick in [0, 5, 10]:
        x = bx0 + tick / 10 * (bx1 - bx0)
        p.append(line(x, 65, x, 282, LIGHT, 1.0, "3 5"))
        p.append(txt(x, 312, str(tick), "tick", "middle"))
    for y, percent in zip(rows, retained):
        width = percent / 10 * (bx1 - bx0)
        p.append(f'<rect x="{bx0}" y="{y-10}" width="{width:.2f}" height="20" fill="{TEAL}"/>')
        p.append(txt(717, y + 7, f"{percent:.1f}%", "label", "middle"))
    p.append(txt(620, 348, "Sparse / full state (%)", "note", "middle"))
    write_svg("fig_srm_tradeoff.svg", p)


def student_figure():
    parent = [17.5297, 17.7279, 19.1275]
    svd = [18.4407, 19.2691, 21.0249]
    diagonal = [18.0642, 18.5744, 20.1656]
    full_noft = [17.5485, 17.7523, 19.1405]
    full_ft = [17.5397, 17.7280, 19.1361]
    delta = lambda values: [a - b for a, b in zip(values, parent)]
    series = [
        ("Weight SVD", delta(svd), BLUE, circle),
        ("Diagonal moment", delta(diagonal), ORANGE, square),
        ("Full moment (No FT)", delta(full_noft), TEAL, triangle),
    ]
    p = svg_open(
        1000, 390, "Low-rank initialization fidelity",
        "Fold-wise MAE increase relative to the parent model on PEMS04 Train-OOF "
        "rank-128 experiments. The right panel magnifies full-moment results near zero.",
    )
    p += [txt(35, 36, "(a) Initialization methods", "heading"),
          txt(660, 36, "(b) Near-parent detail", "heading"),
          line(620, 55, 620, 320, LIGHT, 1.5)]
    rows = [115, 190, 265]
    lx0, lx1 = 160, 585
    xleft = lambda value: lx0 + value / 2.0 * (lx1 - lx0)
    for tick in [0, 0.5, 1.0, 1.5, 2.0]:
        x = xleft(tick)
        p.append(line(x, 65, x, 294, GRAY if tick == 0 else LIGHT,
                      1.5 if tick == 0 else 1.0,
                      None if tick == 0 else "3 5"))
        p.append(txt(x, 323, f"{tick:g}", "tick", "middle"))
    for i, y in enumerate(rows):
        p.append(txt(35, y + 6, f"Fold {i+1}", "label"))
        p.append(line(lx0, y, lx1, y, LIGHT, 0.8))
        for j, (_, values, color, marker) in enumerate(series):
            p.append(marker(xleft(values[i]), y + [-13, 0, 13][j], 6.8, color))

    rx0, rx1 = 710, 950
    xright = lambda value: rx0 + value / 0.05 * (rx1 - rx0)
    for tick in [0, 0.025, 0.05]:
        x = xright(tick)
        p.append(line(x, 65, x, 294, GRAY if tick == 0 else LIGHT,
                      1.5 if tick == 0 else 1.0,
                      None if tick == 0 else "3 5"))
        p.append(txt(x, 323, f"{tick:.3f}" if tick else "0", "tick", "middle"))
    d_noft = delta(full_noft)
    d_ft = delta(full_ft)
    for i, y in enumerate(rows):
        p.append(line(rx0, y, rx1, y, LIGHT, 0.8))
        p.append(triangle(xright(d_noft[i]), y - 10, 7, TEAL))
        p.append(square(xright(d_ft[i]), y + 10, 7, PURPLE))

    legend_y = 365
    legend = [
        (45, "Weight SVD", BLUE, circle),
        (235, "Diagonal moment", ORANGE, square),
        (485, "Full moment (No FT)", TEAL, triangle),
        (790, "Full moment (FT)", PURPLE, square),
    ]
    for x, label, color, marker in legend:
        p.append(marker(x, legend_y - 6, 6, color))
        p.append(txt(x + 14, legend_y, label, "legend"))
    p.append(txt(545, 345, "ΔMAE relative to Parent Model (same fold)", "note", "middle"))
    write_svg("fig_lowrank_initialization.svg", p)


if __name__ == "__main__":
    srm_figure()
    student_figure()
