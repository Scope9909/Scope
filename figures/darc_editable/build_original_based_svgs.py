"""Embed the five supplied PNG figures and make only local, editable corrections.

The source layer is intentionally raster. The overlay elements remain native SVG.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image


SOURCE = Path(os.environ.get(
    "SCOPE_ORIGINAL_FIGURES_DIR", str(Path(__file__).resolve().parent / "source")
))
OUTPUT = Path(__file__).resolve().parent / "original_based"


def text(x, y, content, size=29, color="#102d62", weight="600", anchor=None):
    align = f' text-anchor="{anchor}"' if anchor else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}"{align}>{escape(content)}</text>')


def cover(x, y, width, height, fill="#ffffff", radius=4):
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}"/>')


def note(x, y, width, lines, accent, size=27):
    height = 28 + len(lines) * 36
    result = [f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
              f'rx="12" fill="#ffffff" stroke="{accent}" stroke-width="3"/>']
    result += [text(x + 18, y + 40 + i * 36, line, size) for i, line in enumerate(lines)]
    return "".join(result)


CORRECTIONS = {
    "Figure1_DARC_Motivation.png": {
        "height_extra": 118,
        "items": lambda w, h: [
            note(35, h + 12, w - 70,
                 ["Interpretation: (a) node gains differ; (b) weight similarity does not guarantee output fidelity;",
                  "(c) compression must satisfy prediction, retrieval-decision, and full-file-byte constraints."],
                 "#438bc4", 24),
        ],
    },
    "Figure2_DARC_Overall_Framework.png": {
        "height_extra": 130,
        "items": lambda w, h: [
            note(38, h + 15, w - 76,
                 ["Preparation order: train and freeze Student → select SRM nodes → freeze encoded state → fit Validation calibration.",
                  "Inference: unselected nodes use the Student base prediction; only selected nodes retrieve history."],
                 "#8d5bc0", 34),
        ],
    },
    "Figure3_Low_Rank_Student.png": {
        "height_extra": 0,
        "items": lambda w, h: [
            cover(456, 677, 285, 282, "#ffffff", 10),
            note(459, 686, 278,
                 ["KD loss:", "Teacher vs.", "Student output"], "#d28c67", 21),
            note(459, 831, 278,
                 ["Supervised loss:", "Student output", "vs. train labels"], "#4f86c6", 21),
        ],
    },
    "Figure4_Sparse_Residual_Memory.png": {
        "height_extra": 0,
        "items": lambda w, h: [
            cover(456, 322, 86, 68),
            text(499, 365, "vs.", 35, anchor="middle"),
            cover(539, 231, 370, 47),
            text(548, 265, "Retrieval prediction error", 30),
            cover(94, 557, 447, 112),
            text(106, 597, "Rank gain against state cost;", 27),
            text(106, 637, "freeze the selected node set.", 27),
        ],
    },
    "Figure5_Retrieval_State_Coding.png": {
        "height_extra": 0,
        "items": lambda w, h: [
            cover(501, 321, 153, 87),
            text(517, 359, "Q8 codes", 27),
            text(517, 393, "+ scales", 25),
            cover(683, 321, 225, 87),
            text(697, 359, "Exact values", 27),
            text(697, 393, "(other nodes)", 23),
            cover(977, 553, 235, 102),
            note(977, 556, 238, ["Constraints AND", "mixed file smaller?"], "#e29854", 20),
        ],
    },
}


def main():
    OUTPUT.mkdir(exist_ok=True)
    for name, spec in CORRECTIONS.items():
        original = SOURCE / name
        width, height = Image.open(original).size
        payload = base64.b64encode(original.read_bytes()).decode("ascii")
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height + spec["height_extra"]}" '
            f'viewBox="0 0 {width} {height + spec["height_extra"]}">',
            f'<title>{escape(name.removesuffix(".png"))} — original-based corrections</title>',
            f'<image x="0" y="0" width="{width}" height="{height}" '
            f'href="data:image/png;base64,{payload}"/>',
            '<g font-family="Arial, Aptos, sans-serif">',
            *spec["items"](width, height),
            '</g></svg>',
        ]
        target = OUTPUT / name.replace(".png", ".svg")
        target.write_text("".join(parts), encoding="utf-8")
        print(target, target.stat().st_size)


if __name__ == "__main__":
    main()
