"""Remove only the padded Test rows from the existing PEMS07 teacher dump.

The active data split is ``PEMS07_12to12_index_scaler.npz``. Train and
Validation predictions already match it exactly in length. The legacy dump
included the Test loader's repeated rows used to fill its last batch.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    temporary = output.with_suffix(".tmp.npz")
    if any(path.exists() for path in (output, report_path, temporary)):
        raise FileExistsError("Strict-cache output or temporary file already exists")

    with np.load(args.index, allow_pickle=False) as split:
        lengths = {
            name: len(split[f"idx_{name}"])
            for name in ("train", "val", "test")
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as archive, zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=6, allowZip64=True,
    ) as destination:
        for name in archive.files:
            value = archive[name]
            if name.startswith("teacher_pred_"):
                split_name = name.removeprefix("teacher_pred_")
                if split_name not in lengths:
                    raise ValueError(f"Unexpected teacher split: {name}")
                required = lengths[split_name]
                if split_name in ("train", "val"):
                    if len(value) != required:
                        raise ValueError(f"{name}: {len(value)} != {required}")
                else:
                    excess = len(value) - required
                    if excess < 0 or excess >= 32:
                        raise ValueError(f"Unexpected Test padding: {excess}")
                    if excess and not np.array_equal(
                        value[required:], np.repeat(value[required - 1:required],
                                                    excess, axis=0),
                    ):
                        raise ValueError("Test tail is not repeated batch padding")
                    value = value[:required]
            with destination.open(name + ".npy", "w", force_zip64=True) as stream:
                np.save(stream, value, allow_pickle=False)

    os.replace(temporary, output)
    with np.load(output, allow_pickle=False) as result:
        shapes = {
            name: list(result[f"teacher_pred_{name}"].shape)
            for name in lengths
        }
        if any(shapes[name][0] != lengths[name] for name in lengths):
            raise AssertionError("Strict teacher lengths do not match index")
    report = {
        "source": str(source),
        "index": str(args.index.resolve()),
        "output": str(output),
        "split_lengths": lengths,
        "teacher_shapes": shapes,
        "train_and_val_unchanged": True,
        "test_rows_removed": 24,
        "test_labels_read": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
