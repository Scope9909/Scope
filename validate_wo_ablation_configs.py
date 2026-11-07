"""Validate the dependency graph and flags of PEMS04 w/o ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configurations" / "paper"
EXPERIMENT_ROOT = ROOT / "EXP_experiments" / "TLAST_improved" / "PEMS04"
SEEDS = (1, 2, 3)


def run_id(seed: int, stage: str) -> str:
    return f"paper_pems04_s{seed}_{stage}"


def checkpoint(seed: int, stage: str) -> str:
    return (
        "EXP_experiments/TLAST_improved/PEMS04/"
        f"TLAST_improved_PEMS04_12to12_exp{run_id(seed, stage)}/best.pth"
    )


EXPECTED = {
    "wo_confidence": {
        "parent": "head",
        "lowrank": 1,
        "exact": 1,
        "reranker": 1,
        "confidence": 0,
        "head_finetune": 0,
        "train_reranker": 1,
        "train_confidence": 0,
        "epochs": 20,
    },
    "wo_head_confidence": {
        "parent": "reranker_lowrank",
        "lowrank": 1,
        "exact": 1,
        "reranker": 1,
        "confidence": 1,
        "head_finetune": 0,
        "train_reranker": 0,
        "train_confidence": 1,
        "epochs": 20,
    },
    "wo_head": {
        "parent": "wo_head_confidence",
        "lowrank": 1,
        "exact": 1,
        "reranker": 1,
        "confidence": 1,
        "head_finetune": 0,
        "train_reranker": 1,
        "train_confidence": 1,
        "epochs": 20,
    },
    "wo_reranker_head": {
        "parent": "exact",
        "lowrank": 1,
        "exact": 1,
        "reranker": 0,
        "confidence": 0,
        "head_finetune": 1,
        "train_reranker": 0,
        "train_confidence": 0,
        "epochs": 15,
    },
    "wo_reranker_confidence": {
        "parent": "wo_reranker_head",
        "lowrank": 1,
        "exact": 1,
        "reranker": 0,
        "confidence": 1,
        "head_finetune": 0,
        "train_reranker": 0,
        "train_confidence": 1,
        "epochs": 20,
    },
    "wo_reranker": {
        "parent": "wo_reranker_confidence",
        "lowrank": 1,
        "exact": 1,
        "reranker": 0,
        "confidence": 1,
        "head_finetune": 0,
        "train_reranker": 0,
        "train_confidence": 1,
        "epochs": 20,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_config(seed: int, stage: str, spec: dict) -> None:
    path = CONFIG_DIR / f"pems04_s{seed}_{stage}.json"
    require(path.is_file(), f"missing config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    data = config["data_config"]
    model = config["model_config"]
    training = config["training_config"]

    prefix = f"seed={seed} stage={stage}: "
    require(data["dataset_name"] == "PEMS04", prefix + "wrong dataset")
    require(data["input_length"] == 12, prefix + "wrong input length")
    require(data["predict_length"] == 12, prefix + "wrong horizon")
    require(model["use_analog_retrieval_residual"] == 0, prefix + "legacy retrieval enabled")
    require(
        model["use_analog_lowrank_retrieval_residual"] == spec["lowrank"],
        prefix + "low-rank flag mismatch",
    )
    require(
        model["use_analog_exact_match_residual"] == spec["exact"],
        prefix + "exact flag mismatch",
    )
    require(
        model["use_analog_candidate_reranker"] == spec["reranker"],
        prefix + "reranker flag mismatch",
    )
    require(
        model["use_analog_confidence_aware_retrieval_residual"]
        == spec["confidence"],
        prefix + "confidence flag mismatch",
    )
    require(
        model["retrieval_aware_head_finetune"] == spec["head_finetune"],
        prefix + "head-finetune flag mismatch",
    )
    require(
        model["freeze_parent_train_analog_candidate_reranker"]
        == spec["train_reranker"],
        prefix + "reranker training flag mismatch",
    )
    require(
        model["freeze_parent_train_analog_confidence_aware_retrieval"]
        == spec["train_confidence"],
        prefix + "confidence training flag mismatch",
    )
    require(
        model["frozen_parent_ckpt"] == checkpoint(seed, spec["parent"]),
        prefix + "wrong parent checkpoint",
    )
    require(training["epochs"] == spec["epochs"], prefix + "wrong epoch budget")
    require(training["device"] == "cuda:0", prefix + "wrong device")

    parent = ROOT / model["frozen_parent_ckpt"]
    if not spec["parent"].startswith("wo_"):
        require(parent.is_file(), prefix + f"missing existing parent: {parent}")


def validate_existing_checkpoints() -> None:
    import torch

    required_fragments = {
        "exact": (
            "analog_lowrank_retrieval_residual",
            "analog_exact_match_residual",
        ),
        "reranker_lowrank": (
            "analog_lowrank_retrieval_residual",
            "analog_exact_match_residual",
            "analog_candidate_reranker",
        ),
        "head": (
            "analog_lowrank_retrieval_residual",
            "analog_exact_match_residual",
            "analog_candidate_reranker",
            "time_wise_predictor",
        ),
    }
    for seed in SEEDS:
        for stage, fragments in required_fragments.items():
            path = ROOT / checkpoint(seed, stage)
            state = torch.load(path, map_location="cpu")
            require(isinstance(state, dict), f"unexpected checkpoint type: {path}")
            keys = tuple(state)
            for fragment in fragments:
                require(
                    any(fragment in key for key in keys),
                    f"{path} lacks required module {fragment}",
                )
            for key, value in state.items():
                if torch.is_tensor(value):
                    require(
                        bool(torch.isfinite(value).all()),
                        f"non-finite checkpoint tensor: {path}:{key}",
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-checkpoints", action="store_true")
    args = parser.parse_args()
    for seed in SEEDS:
        for stage, spec in EXPECTED.items():
            validate_config(seed, stage, spec)
    if args.check_checkpoints:
        validate_existing_checkpoints()
    print(f"Validated {len(SEEDS) * len(EXPECTED)} w/o configs and dependency edges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
