"""Generate isolated, shorter PEMS-BAY 70/10/20 experiment configs."""

from __future__ import annotations

import json
from pathlib import Path

from make_cross_dataset_experiment_configs import STAGES, make_stage


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'configurations' / 'cross_dataset_712'
DATASET = 'PEMSBAY'
INDEX_CACHE = 'datasets/PEMSBAY/PEMSBAY_12to12_index_scaler_712.npz'
RETRIEVAL_CACHE = (
    'datasets/PEMSBAY/PEMSBAY_12to12_analog_k8_exact_mean_712.npz'
)
CANDIDATE_CACHE = (
    'datasets/PEMSBAY/PEMSBAY_12to12_analog_k8_candidate_indices_712.npz'
)
EPOCHS = {
    'backbone': 80,
    'lowrank': 10,
    'exact': 20,
    'reranker_lowrank': 20,
    'head': 8,
    'confidence': 15,
    'final_joint': 20,
}


def run_id(stage: str) -> str:
    return f'paper_pemsbay712_s1_{stage}'


def checkpoint(stage: str) -> str:
    return (
        'EXP_experiments/TLAST_improved/PEMSBAY/'
        f'TLAST_improved_PEMSBAY_12to12_exp{run_id(stage)}/best.pth'
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = {
        'lowrank': 'backbone',
        'exact': 'lowrank',
        'reranker_lowrank': 'exact',
        'head': 'reranker_lowrank',
        'confidence': 'head',
        'final_joint': 'confidence',
    }
    for stage in STAGES:
        config = make_stage(DATASET, stage)
        data = config['data_config']
        data['train_ratio'] = 0.7
        data['test_ratio'] = 0.2
        data['index_scaler_path'] = INDEX_CACHE

        model = config['model_config']
        model['analog_retrieval_cache'] = RETRIEVAL_CACHE
        model['analog_candidate_cache'] = CANDIDATE_CACHE
        if stage != 'backbone':
            model['frozen_parent_ckpt'] = checkpoint(previous[stage])

        config['training_config']['epochs'] = EPOCHS[stage]
        target = OUTPUT_DIR / f'pemsbay712_s1_{stage}.json'
        with target.open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(config, handle, indent=2)
            handle.write('\n')
        print(f'Generated {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
