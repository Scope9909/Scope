"""Copy the four hash-pinned PEMS04 training assets from a trusted workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training" / "pems04.json"

SOURCE_PATHS = {
    "datasets/PEMS04/training_assets/analog_k4.npz":
        "datasets/PEMS04/PEMS04_12to12_analog_k4.npz",
    "datasets/PEMS04/training_assets/analog_k4_exact_mean.npz":
        "datasets/PEMS04/PEMS04_12to12_analog_k4_exact_mean.npz",
    "datasets/PEMS04/training_assets/predictor_activation_second_moment.npz":
        "paper_results/PEMS04_parent_predictor_activation_second_moment_s1.npz",
    "datasets/PEMS04/training_assets/teacher_predictions_joint3.npz":
        "datasets/PEMS04/PEMS04_12to12_parent_predictions_joint3_strict.npz",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--apply", action="store_true",
        help="Copy after verification. Without this flag, only preview and verify.",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    source_root = cli.source.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    expected = json.loads(CONFIG.read_text(encoding="utf-8"))["training_assets"]
    rows = []
    for destination_name, source_name in SOURCE_PATHS.items():
        source = source_root / source_name
        destination = ROOT / destination_name
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = digest(source)
        if actual != expected[destination_name]:
            raise ValueError(
                f"Source hash mismatch for {source_name}: {actual}; "
                f"expected {expected[destination_name]}"
            )
        if destination.exists():
            destination_hash = digest(destination)
            if destination_hash != actual:
                raise FileExistsError(
                    f"Refusing to replace mismatching destination: {destination}"
                )
            action = "already present"
        elif cli.apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if digest(destination) != actual:
                raise IOError(f"Post-copy hash check failed: {destination}")
            action = "copied"
        else:
            action = "would copy"
        rows.append({
            "source": str(source),
            "destination": str(destination.relative_to(ROOT)),
            "bytes": source.stat().st_size,
            "sha256": actual,
            "action": action,
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
