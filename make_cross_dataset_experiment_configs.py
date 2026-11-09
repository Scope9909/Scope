"""Generate Scope retrieval-augmented cross-dataset experiment configs.

The generated dependency chain is:
backbone -> lowrank -> exact -> reranker_lowrank -> head -> confidence
-> final_joint.  Cache creation is handled by run_cross_dataset_experiments.ps1
after the backbone has created the canonical time/index-scaler files.

The seed is a CLI argument (`--seed`, default 1).  For a given dataset the
generated JSON body is seed-independent; only the stage `run_id`s / checkpoint
paths embed the seed, so `--seed 1` reproduces the original filenames exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configurations"
OUTPUT_DIR = CONFIG_ROOT / "cross_dataset"
SUPPORTED_DATASETS = ("PEMS03", "PEMS07", "PEMS08", "PEMSBAY")
STAGES = (
    "backbone",
    "lowrank",
    "exact",
    "reranker_lowrank",
    "head",
    "confidence",
    "final_joint",
)

# Overridden in main() from --seed.  Kept module-level so run_id/checkpoint stay
# single-argument helpers (downstream code calls them with (dataset, stage)).
SEED = 1


def slug(dataset: str) -> str:
    return dataset.lower()


def run_id(dataset: str, stage: str) -> str:
    return f"paper_{slug(dataset)}_s{SEED}_{stage}"


def checkpoint(dataset: str, stage: str) -> str:
    return (
        f"EXP_experiments/TLAST_improved/{dataset}/"
        f"TLAST_improved_{dataset}_12to12_exp{run_id(dataset, stage)}/best.pth"
    )


def retrieval_cache(dataset: str) -> str:
    return f"datasets/{dataset}/{dataset}_12to12_analog_k8_exact_mean.npz"


def candidate_cache(dataset: str) -> str:
    return (
        f"datasets/{dataset}/"
        f"{dataset}_12to12_analog_k8_candidate_indices.npz"
    )


def reset_retrieval_flags(model: dict, dataset: str) -> None:
    model.update(
        {
            "use_analog_retrieval_residual": 0,
            "use_analog_lowrank_retrieval_residual": 0,
            "analog_lowrank_rank": 2,
            "analog_retrieval_max": 1.0,
            "analog_retrieval_k": 8,
            "analog_retrieval_aggregation": "mean",
            "analog_retrieval_cache": retrieval_cache(dataset),
            "use_analog_exact_match_residual": 0,
            "analog_exact_match_max": 1.0,
            "analog_exact_min_valid_history": 6,
            "use_analog_candidate_reranker": 0,
            "analog_candidate_feature_dim": 8,
            "analog_candidate_disagreement_dim": 2,
            "analog_candidate_hidden": 16,
            "analog_candidate_min_valid_history": 6,
            "analog_candidate_cache": candidate_cache(dataset),
            "use_analog_confidence_aware_retrieval_residual": 0,
            "analog_confidence_max": 0.25,
            "analog_confidence_node_rank": 0,
            "analog_confidence_feature_dim": 3,
            "freeze_parent_train_analog_retrieval": 0,
            "freeze_parent_train_analog_lowrank_retrieval": 0,
            "freeze_parent_train_analog_exact_match": 0,
            "freeze_parent_train_analog_candidate_reranker": 0,
            "joint_train_lowrank_with_reranker": 0,
            "freeze_parent_train_analog_confidence_aware_retrieval": 0,
            "freeze_parent_train_analog_joint_retrieval": 0,
            "retrieval_aware_head_finetune": 0,
            "retrieval_aware_full_predictor": 0,
            "frozen_parent_ckpt": "none",
        }
    )


def set_training(
    training: dict,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    training.update(
        {
            "device": "cuda:0",
            "batch_size": 32,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "print_every": 50,
            "early_stop": False,
            "early_stop_step": 20,
            "lr_decay": 1,
            "loss_type": "masked_mae_0",
            "huber_delta": 2,
            "eval_mask": 0,
            "lr_scheduler_type": "cosinelr",
        }
    )


def base_config(dataset: str) -> dict:
    with (CONFIG_ROOT / "ablation_minus_ftle.json").open(
        "r", encoding="utf-8"
    ) as handle:
        config = json.load(handle)
    config["data_config"]["dataset_name"] = dataset
    reset_retrieval_flags(config["model_config"], dataset)
    set_training(
        config["training_config"],
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-3,
    )
    return config


def make_stage(dataset: str, stage: str) -> dict:
    config = copy.deepcopy(base_config(dataset))
    model = config["model_config"]
    training = config["training_config"]

    if stage == "backbone":
        return config

    model["use_analog_lowrank_retrieval_residual"] = 1
    if stage == "lowrank":
        model.update(
            {
                "freeze_parent_train_analog_lowrank_retrieval": 1,
                "frozen_parent_ckpt": checkpoint(dataset, "backbone"),
            }
        )
        set_training(training, epochs=20, learning_rate=1e-3, weight_decay=1e-3)
    elif stage == "exact":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "freeze_parent_train_analog_exact_match": 1,
                "frozen_parent_ckpt": checkpoint(dataset, "lowrank"),
            }
        )
        set_training(training, epochs=20, learning_rate=1e-3, weight_decay=1e-3)
    elif stage == "reranker_lowrank":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "freeze_parent_train_analog_candidate_reranker": 1,
                "joint_train_lowrank_with_reranker": 1,
                "frozen_parent_ckpt": checkpoint(dataset, "exact"),
            }
        )
        set_training(training, epochs=50, learning_rate=1e-3, weight_decay=1e-3)
    elif stage == "head":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "retrieval_aware_head_finetune": 1,
                "retrieval_aware_full_predictor": 0,
                "frozen_parent_ckpt": checkpoint(dataset, "reranker_lowrank"),
            }
        )
        set_training(training, epochs=15, learning_rate=2e-5, weight_decay=1e-4)
    elif stage == "confidence":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "use_analog_confidence_aware_retrieval_residual": 1,
                "analog_confidence_feature_dim": 3,
                "analog_confidence_node_rank": 0,
                "freeze_parent_train_analog_confidence_aware_retrieval": 1,
                "frozen_parent_ckpt": checkpoint(dataset, "head"),
            }
        )
        set_training(training, epochs=20, learning_rate=1e-3, weight_decay=1e-4)
    elif stage == "final_joint":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "use_analog_confidence_aware_retrieval_residual": 1,
                "analog_confidence_feature_dim": 6,
                "analog_confidence_node_rank": 2,
                "freeze_parent_train_analog_candidate_reranker": 1,
                "freeze_parent_train_analog_confidence_aware_retrieval": 1,
                "frozen_parent_ckpt": checkpoint(dataset, "confidence"),
            }
        )
        set_training(training, epochs=20, learning_rate=1e-3, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Training seed.  Only stage run_ids / ckpt paths depend on it.",
    )
    return parser.parse_args()


def main() -> int:
    global SEED
    args = parse_args()
    SEED = args.seed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for dataset in args.datasets:
        raw_data = ROOT / "datasets" / dataset / "data.npz"
        if not raw_data.exists():
            raise FileNotFoundError(f"Missing raw dataset: {raw_data}")
        for stage in STAGES:
            target = OUTPUT_DIR / f"{slug(dataset)}_s{SEED}_{stage}.json"
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(make_stage(dataset, stage), handle, indent=2)
                handle.write("\n")
            generated.append(target)
    print(f"Generated {len(generated)} configs (seed={SEED}) under {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
