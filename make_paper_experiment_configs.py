"""Generate a reproducible PEMS04 paper experiment pipeline.

The generated configs reuse the already trained seed-specific TLAST backbone
checkpoints, then rebuild every retrieval stage with one declared protocol.
No training is started by this script.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configurations"
OUTPUT_DIR = CONFIG_ROOT / "paper"

BACKBONE_BY_SEED = {
    1: (
        "EXP_experiments/TLAST_improved/PEMS04/"
        "TLAST_improved_PEMS04_12to12_exp07170128_s1/best.pth"
    ),
    2: (
        "EXP_experiments/TLAST_improved/PEMS04/"
        "TLAST_improved_PEMS04_12to12_exp07181424_s2/best.pth"
    ),
    3: (
        "EXP_experiments/TLAST_improved/PEMS04/"
        "TLAST_improved_PEMS04_12to12_exp07181621_s3/best.pth"
    ),
}

RETRIEVAL_CACHE = (
    "datasets/PEMS04/PEMS04_12to12_analog_k8_exact_mean.npz"
)
CANDIDATE_CACHE = (
    "datasets/PEMS04/PEMS04_12to12_analog_k8_candidate_indices.npz"
)


def run_id(seed: int, stage: str) -> str:
    return f"paper_pems04_s{seed}_{stage}"


def checkpoint(seed: int, stage: str) -> str:
    return (
        "EXP_experiments/TLAST_improved/PEMS04/"
        "TLAST_improved_PEMS04_12to12_exp"
        f"{run_id(seed, stage)}/best.pth"
    )


def reset_retrieval_flags(model: dict) -> None:
    model.update(
        {
            "use_analog_retrieval_residual": 0,
            "use_analog_lowrank_retrieval_residual": 1,
            "analog_lowrank_rank": 2,
            "analog_retrieval_max": 1.0,
            "analog_retrieval_k": 8,
            "analog_retrieval_aggregation": "mean",
            "analog_retrieval_cache": RETRIEVAL_CACHE,
            "use_analog_exact_match_residual": 0,
            "analog_exact_match_max": 1.0,
            "analog_exact_min_valid_history": 6,
            "use_analog_candidate_reranker": 0,
            "use_analog_direct_prediction": 0,
            "analog_candidate_feature_dim": 8,
            "analog_candidate_disagreement_dim": 2,
            "analog_candidate_hidden": 16,
            "analog_candidate_min_valid_history": 6,
            "analog_candidate_cache": CANDIDATE_CACHE,
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


def base_config() -> dict:
    path = CONFIG_ROOT / "ablation_minus_ftle.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    reset_retrieval_flags(config["model_config"])
    return config


def make_stage(base: dict, seed: int, stage: str) -> dict:
    config = copy.deepcopy(base)
    model = config["model_config"]
    training = config["training_config"]

    if stage == "lowrank":
        model.update(
            {
                "freeze_parent_train_analog_lowrank_retrieval": 1,
                "frozen_parent_ckpt": BACKBONE_BY_SEED[seed],
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-3
        )
    elif stage == "exact":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "freeze_parent_train_analog_exact_match": 1,
                "frozen_parent_ckpt": checkpoint(seed, "lowrank"),
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-3
        )
    elif stage in {"reranker_only", "reranker_lowrank"}:
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "freeze_parent_train_analog_candidate_reranker": 1,
                "joint_train_lowrank_with_reranker": int(
                    stage == "reranker_lowrank"
                ),
                "frozen_parent_ckpt": checkpoint(seed, "exact"),
            }
        )
        set_training(
            training, epochs=50, learning_rate=1e-3, weight_decay=1e-3
        )
    elif stage == "direct_retrieval":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "use_analog_direct_prediction": 1,
                "freeze_parent_train_analog_candidate_reranker": 1,
                "frozen_parent_ckpt": checkpoint(seed, "exact"),
            }
        )
        set_training(
            training, epochs=50, learning_rate=1e-3, weight_decay=1e-3
        )
    elif stage == "head":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "retrieval_aware_head_finetune": 1,
                "retrieval_aware_full_predictor": 0,
                "frozen_parent_ckpt": checkpoint(seed, "reranker_lowrank"),
            }
        )
        set_training(
            training, epochs=15, learning_rate=2e-5, weight_decay=1e-4
        )
    elif stage == "confidence":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "use_analog_confidence_aware_retrieval_residual": 1,
                "analog_confidence_feature_dim": 3,
                "analog_confidence_node_rank": 0,
                "freeze_parent_train_analog_confidence_aware_retrieval": 1,
                "frozen_parent_ckpt": checkpoint(seed, "head"),
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-4
        )
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
                "frozen_parent_ckpt": checkpoint(seed, "confidence"),
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-4
        )
    elif stage == "final_direct_joint":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "use_analog_confidence_aware_retrieval_residual": 1,
                "analog_confidence_feature_dim": 6,
                "analog_confidence_node_rank": 2,
                "freeze_parent_train_analog_candidate_reranker": 1,
                "freeze_parent_train_analog_confidence_aware_retrieval": 1,
                "frozen_parent_ckpt": checkpoint(seed, "head"),
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-4
        )
    elif stage == "wo_confidence":
        # Full pipeline through the retrieval-aware head, followed by the
        # same 20-epoch final refinement budget with confidence disabled.
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "freeze_parent_train_analog_candidate_reranker": 1,
                "frozen_parent_ckpt": checkpoint(seed, "head"),
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-4
        )
    elif stage == "wo_head_confidence":
        # Confidence warm-up directly from the reranker parent: the only
        # omitted operation is retrieval-aware head fine-tuning.
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "use_analog_confidence_aware_retrieval_residual": 1,
                "analog_confidence_feature_dim": 3,
                "analog_confidence_node_rank": 0,
                "freeze_parent_train_analog_confidence_aware_retrieval": 1,
                "frozen_parent_ckpt": checkpoint(seed, "reranker_lowrank"),
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-4
        )
    elif stage == "wo_head":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_candidate_reranker": 1,
                "use_analog_confidence_aware_retrieval_residual": 1,
                "analog_confidence_feature_dim": 6,
                "analog_confidence_node_rank": 2,
                "freeze_parent_train_analog_candidate_reranker": 1,
                "freeze_parent_train_analog_confidence_aware_retrieval": 1,
                "frozen_parent_ckpt": checkpoint(seed, "wo_head_confidence"),
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-4
        )
    elif stage == "wo_reranker_head":
        # The retrieval-aware head supports the low-rank + exact parent even
        # when candidate reranking is absent.
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "retrieval_aware_head_finetune": 1,
                "retrieval_aware_full_predictor": 0,
                "frozen_parent_ckpt": checkpoint(seed, "exact"),
            }
        )
        set_training(
            training, epochs=15, learning_rate=2e-5, weight_decay=1e-4
        )
    elif stage == "wo_reranker_confidence":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_confidence_aware_retrieval_residual": 1,
                "analog_confidence_feature_dim": 3,
                "analog_confidence_node_rank": 0,
                "freeze_parent_train_analog_confidence_aware_retrieval": 1,
                "frozen_parent_ckpt": checkpoint(seed, "wo_reranker_head"),
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-4
        )
    elif stage == "wo_reranker":
        model.update(
            {
                "use_analog_exact_match_residual": 1,
                "use_analog_confidence_aware_retrieval_residual": 1,
                "analog_confidence_feature_dim": 6,
                "analog_confidence_node_rank": 2,
                "freeze_parent_train_analog_confidence_aware_retrieval": 1,
                "frozen_parent_ckpt": checkpoint(
                    seed, "wo_reranker_confidence"
                ),
            }
        )
        set_training(
            training, epochs=20, learning_rate=1e-3, weight_decay=1e-4
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1],
        choices=sorted(BACKBONE_BY_SEED),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stages = (
        "lowrank",
        "exact",
        "reranker_only",
        "reranker_lowrank",
        "direct_retrieval",
        "head",
        "confidence",
        "final_joint",
        "final_direct_joint",
        "wo_confidence",
        "wo_head_confidence",
        "wo_head",
        "wo_reranker_head",
        "wo_reranker_confidence",
        "wo_reranker",
    )
    template = base_config()
    generated = []
    for seed in args.seeds:
        for stage in stages:
            target = OUTPUT_DIR / f"pems04_s{seed}_{stage}.json"
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    make_stage(template, seed, stage),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
                handle.write("\n")
            generated.append(target)

    for required in (
        ROOT / RETRIEVAL_CACHE,
        ROOT / CANDIDATE_CACHE,
        *(ROOT / BACKBONE_BY_SEED[seed] for seed in args.seeds),
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required experiment input is missing: {required}")

    print(f"Generated {len(generated)} configs under {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
