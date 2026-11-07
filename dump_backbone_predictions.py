"""dump_backbone_predictions.py —— dump joint3 checkpoint 的 backbone-only 前向预测.

目的: candidate-relative residual construction 实验需要每个 train endpoint 的
parent(base) 预测 Ŷ_i.  部署流水线中检索残差修正的对象是 lowrank 阶段的
main_output(= backbone 输出), 因此 Ŷ_i 取 backbone 对候选历史的预测, 与被修正
的 P 同源、口径一致.

做法: 复制 joint3 config, 关闭全部 analog adapter 开关(模块不构建, 但它们都在
backbone 之后构建, 不影响 backbone 权重名), strict=False 加载 joint3 best.pth:
  - missing keys 必须为空(backbone 权重全部命中)
  - unexpected keys 必须全部以 'analog_' 开头(只是被跳过的 adapter)
然后在 train/val/test 上前向, dump z-score 空间预测.

输出 npz 结构(与 dump_teacher_predictions.py 对齐):
  backbone_pred_train/val/test: (num_samples, predict_length, num_nodes, 1) float32 (z 空间)
  scaler_mean / scaler_std
"""
import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch

ANALOG_FLAGS_TO_DISABLE = (
    'use_analog_retrieval_residual',
    'use_analog_lowrank_retrieval_residual',
    'use_analog_exact_match_residual',
    'use_analog_candidate_reranker',
    'use_analog_confidence_aware_retrieval_residual',
    'use_analog_query_conditioned_residual',
    'use_analog_correction_utility_gate',
    'use_analog_adaptive_residual_mixer',
    'use_analog_weekly_retrieval_residual',
    'use_analog_metric_retrieval_residual',
    'use_analog_node_horizon_retrieval_residual',
    'use_analog_temporal_gap_residual',
    'use_analog_temporal_lowrank_decoder',
    'use_analog_code_domain_aggregation',
    'use_analog_direct_prediction',
    'use_analog_candidate_reranker',
)


def _setup_logger():
    logger = logging.getLogger('dump_backbone')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_dir',
        default=(
            'EXP_experiments/TLAST_improved/PEMS04/'
            'TLAST_improved_PEMS04_12to12_exppaper_pems04_s1_latc_pca_r8_joint3'
        ),
    )
    parser.add_argument(
        '--output',
        default='datasets/PEMS04/PEMS04_12to12_backbone_predictions_joint3.npz',
    )
    args = parser.parse_args()

    logger = _setup_logger()
    exp_dir = os.path.abspath(args.exp_dir)
    best_pth = os.path.join(exp_dir, 'best.pth')
    config_pkl = os.path.join(exp_dir, 'config.pkl')

    sys_argv_save = sys.argv
    sys.argv = sys.argv[:1]
    from utils import datasets_utils
    from utils.datasets_utils import DataLoader_time
    from run import trainer
    sys.argv = sys_argv_save

    with open(config_pkl, 'rb') as f:
        run_args = pickle.load(f)

    # Disable EVERY analog switch generically so no adapter module is built and
    # no retrieval payload is fed to forward.  This is stricter than an
    # explicit flag whitelist, which could miss a newly added adapter.
    disabled = []
    for flag in sorted(vars(run_args)):
        if flag.startswith('use_analog') and getattr(run_args, flag, 0):
            setattr(run_args, flag, 0)
            disabled.append(flag)
    for cache_flag in (
        'analog_candidate_cache',
        'analog_candidate_code_cache',
        'analog_retrieval_cache',
        'analog_metric_retrieval_cache',
        'analog_weekly_retrieval_cache',
    ):
        if hasattr(run_args, cache_flag):
            setattr(run_args, cache_flag, 'none')
    logger.info(f'Disabled analog flags: {disabled}')

    run_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_args.cache_data_path = 'datasets'
    if isinstance(getattr(run_args, 'k_hops', '1,2'), str):
        run_args.k_hops = [int(v) for v in run_args.k_hops.split(',')]

    pearson_path = getattr(run_args, 'pearson_adj_path', None)
    if pearson_path and os.path.exists(pearson_path):
        run_args.pearson_adj = torch.FloatTensor(np.load(pearson_path))
    elif os.path.exists(
        f"datasets/{run_args.dataset_name}/{run_args.dataset_name}_pearson_adj.npy"
    ):
        run_args.pearson_adj = torch.FloatTensor(np.load(
            f"datasets/{run_args.dataset_name}/{run_args.dataset_name}_pearson_adj.npy"
        ))
    else:
        run_args.pearson_adj = None

    device = torch.device(run_args.device)
    dataloader, num_nodes, scaler, _ = datasets_utils.load_dataset(
        run_args, logger
    )
    run_args.num_nodes = num_nodes

    engine = trainer(scaler=scaler, device=device, args=run_args, inverse=False)
    state = torch.load(best_pth, map_location=device)
    result = engine.model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing:
        raise RuntimeError(f'missing keys (backbone weights not in ckpt): {missing[:20]}')
    bad = [k for k in unexpected if not k.startswith('analog_')]
    if bad:
        raise RuntimeError(f'unexpected non-analog keys: {bad[:20]}')
    logger.info(
        f'load ok: {len(unexpected)} analog adapter keys skipped, '
        f'0 missing, 0 non-analog unexpected'
    )
    engine.model.eval()

    sample_counts = {
        k: dataloader[f'{k}_loader'].size for k in ('train', 'val', 'test')
    }
    logger.info(f'Loader sizes: {sample_counts}')

    # Split ground-truth lengths from the index file.  The train loader uses
    # drop_last=True (tail samples lost) and the test loader pads with
    # repeated last rows; we repair both so dump rows align 1:1 with
    # idx_train / idx_val / idx_test.
    # NOTE: DataLoader_time does not shuffle on its own (shuffle() is an
    # explicit external call), so iterating get_iterator() emits rows in idx
    # order -- positional alignment with idx_* is exact.
    index_npz = np.load(
        f"datasets/{run_args.dataset_name}/"
        f"{run_args.dataset_name}_{run_args.input_length}to"
        f"{run_args.predict_length}_index_scaler.npz"
    )
    split_idx = {
        'train': np.asarray(index_npz['idx_train']),
        'val': np.asarray(index_npz['idx_val']),
        'test': np.asarray(index_npz['idx_test']),
    }
    index_npz.close()
    for k, v in split_idx.items():
        logger.info(f'idx_{k}: len={len(v)}')

    # Reuse the very same arrays the loaders were built with.
    ref_loader = dataloader['train_loader']
    shared_data = ref_loader.data
    shared_time = ref_loader.time_data
    shared_scaler = ref_loader.scaler

    output = {}
    for split in ('train', 'val', 'test'):
        preds = []
        with torch.no_grad():
            loader = dataloader[f'{split}_loader']
            for batch in loader.get_iterator():
                model_inputs, (y, y_time) = batch
                x = torch.Tensor(model_inputs[0]).to(device)
                x_time = torch.LongTensor(model_inputs[1]).to(device)
                y_t = torch.LongTensor(y_time).to(device)
                out, _ = engine.model([x, x_time, y_t])
                preds.append(out.cpu())
        arr = torch.cat(preds, dim=0).numpy().astype(np.float32)
        target_len = len(split_idx[split])
        logger.info(
            f'{split}: loader rows={len(arr)}, target rows={target_len}'
        )
        if split == 'train' and len(arr) < target_len:
            # drop_last dropped the tail; forward those positions exactly
            # with a small dedicated loader (no scaler/padding tricks).
            tail_idx = split_idx[split][len(arr):]
            logger.info(f'forwarding dropped train tail: {len(tail_idx)} rows')
            tail_loader = DataLoader_time(
                shared_data, shared_time, tail_idx,
                run_args.input_length, run_args.predict_length,
                len(tail_idx), logger, scaler=shared_scaler,
                drop_last=False,
            )
            with torch.no_grad():
                for batch in tail_loader.get_iterator():
                    model_inputs, (y, y_time) = batch
                    x = torch.Tensor(model_inputs[0]).to(device)
                    x_time = torch.LongTensor(model_inputs[1]).to(device)
                    y_t = torch.LongTensor(y_time).to(device)
                    out, _ = engine.model([x, x_time, y_t])
                    preds.append(out.cpu())
            arr = torch.cat(preds, dim=0).numpy().astype(np.float32)
        if len(arr) != target_len:
            # test loader pads with repeated last rows: truncate them.
            if len(arr) > target_len:
                arr = arr[:target_len]
            else:
                raise RuntimeError(
                    f'{split}: got {len(arr)} rows, expected {target_len}'
                )
        if not np.isfinite(arr).all():
            raise RuntimeError(f'{split} predictions contain non-finite values')
        output[f'backbone_pred_{split}'] = arr
        logger.info(f'{split} pred shape={arr.shape} (aligned to idx_{split})')

    output['scaler_mean'] = scaler.mean if hasattr(scaler, 'mean') else np.zeros(1)
    output['scaler_std'] = scaler.std if hasattr(scaler, 'std') else np.ones(1)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path + '.tmp.npz'
    np.savez_compressed(tmp, **output)
    os.replace(tmp, output_path)
    logger.info(f'Saved backbone predictions to {output_path}')


if __name__ == '__main__':
    main()
