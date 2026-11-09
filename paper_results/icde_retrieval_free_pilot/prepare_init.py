"""Create an isolated retrieval-free initialization from the PEMS04 parent.

The dense predictor and backbone tensors are copied unchanged. Only the four
retrieval-adapter tensors absent from the retrieval-free architecture are
removed. The source checkpoint is never modified.
"""

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT / "EXP_experiments/TLAST_improved/PEMS04"
    / "TLAST_improved_PEMS04_12to12_exp07181357_s1/best.pth"
)
OUTPUT = Path(__file__).resolve().parent / "parent_without_retrieval_adapters_s1.pth"
RETRIEVAL_PREFIXES = (
    "analog_lowrank_retrieval_residual.",
    "analog_exact_match_residual.",
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    state = torch.load(SOURCE, map_location="cpu", weights_only=True)
    removed = [
        key for key in state
        if key.startswith(RETRIEVAL_PREFIXES)
    ]
    if len(removed) != 4:
        raise ValueError(f"Expected exactly four retrieval tensors, got {removed}")
    stripped = type(state)(
        (key, value) for key, value in state.items() if key not in removed
    )
    torch.save(stripped, OUTPUT, _use_new_zipfile_serialization=False)
    print(f"Source: {SOURCE}")
    print(f"Output: {OUTPUT}")
    print(f"Removed: {removed}")
    print(f"Retained tensors: {len(stripped)}")


if __name__ == "__main__":
    main()
