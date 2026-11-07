"""
dump_teacher_predictions.py —— 离线 dump teacher 模型在 train/val/test 的前向预测,
存为 npz, 用作知识蒸馏的 soft target.

用法:
  python dump_teacher_predictions.py --exp_dir EXP_.../..._exp07181357_s1 --output datasets/PEMS04/PEMS04_12to12_teacher_predictions.npz

输出 npz 结构（可由 --splits 限制，以避免 Validation-only 实验读取 Test）:
  teacher_pred_train: shape (num_train_samples, predict_length, num_nodes, in_dim), float32
  teacher_pred_val:   shape (num_val_samples, ...)
  teacher_pred_test:  shape (num_test_samples, ...)
  scaler_mean / scaler_std: 用于反归一化（与 student 对齐 metric 用）
预测在 z-score 空间（与训练 loss 同空间）。
"""
import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def _setup_logger():
    logger = logging.getLogger('dump_teacher')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    return logger


def _resolve_exp_dir(p):
    direct = Path(p).expanduser().resolve()
    if direct.is_dir() and (direct / 'best.pth').is_file():
        return str(direct)
    # Backward-compatible convenience for passing only an experiment folder
    # name, but do not hard-code a dataset or model family.
    matches = [
        candidate.parent.resolve()
        for candidate in Path('EXP_experiments').rglob('best.pth')
        if candidate.parent.name == Path(p).name
    ]
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise RuntimeError(
            f'Ambiguous experiment directory {p!r}; matches={matches}'
        )
    raise FileNotFoundError(f'best.pth not found near {direct}')


def _forward_split(engine, dataloader, split_name, num_samples, device):
    preds = []
    loader = dataloader[f'{split_name}_loader']
    # The training loader normally drops its final partial batch.  A teacher
    # cache, however, must contain one prediction for every sample so that KD
    # targets remain positionally aligned after shuffling.  Temporarily make
    # every split exhaustive and restore the loader state afterwards.
    original_drop_last = loader.drop_last
    original_num_batch = loader.num_batch
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    try:
        with torch.no_grad():
            for batch in loader.get_iterator():
                model_inputs, (y, y_time) = batch
                # model_inputs = [x, x_time, retrieval?, exact_retrieval?, exact_available?]
                x = torch.Tensor(model_inputs[0]).to(device)
                x_time = torch.LongTensor(model_inputs[1]).to(device)
                y_t = torch.LongTensor(y_time).to(device)
                extras = []
                for arr in model_inputs[2:]:
                    t = torch.as_tensor(
                        arr,
                        dtype=(
                            torch.float16
                            if hasattr(arr, 'dtype') and arr.dtype == np.float16
                            else torch.float32
                        ),
                        device=device,
                    )
                    extras.append(t)
                out, _ = engine.model([x, x_time, y_t] + extras)
                preds.append(out.cpu())
    finally:
        loader.drop_last = original_drop_last
        loader.num_batch = original_num_batch
    out = torch.cat(preds, dim=0)[:num_samples].numpy()
    if len(out) != num_samples:
        raise RuntimeError(
            f'{split_name} teacher dump is incomplete: '
            f'{len(out)} != {num_samples}'
        )
    if not np.isfinite(out).all():
        raise RuntimeError(
            f'{split_name} teacher dump contains non-finite values'
        )
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_dir',
        required=True,
        help='teacher 实验目录（含 config.pkl + best.pth）',
    )
    parser.add_argument(
        '--output',
        default='datasets/PEMS04/PEMS04_12to12_teacher_predictions.npz',
    )
    parser.add_argument(
        '--teacher-name',
        default=None,
        help='Human-readable teacher identifier stored in the output cache.',
    )
    parser.add_argument(
        '--splits',
        default='train,val,test',
        help='Comma-separated splits to export: train,val,test.',
    )
    parser.add_argument(
        '--retrieval-cache-override',
        default=None,
        help=(
            'Optional retrieval-cache replacement for storage/quantization '
            'audits; the saved experiment config is not modified.'
        ),
    )
    args = parser.parse_args()

    requested_splits = tuple(
        split.strip().lower()
        for split in args.splits.split(',')
        if split.strip()
    )
    invalid_splits = sorted(
        set(requested_splits).difference({'train', 'val', 'test'})
    )
    if not requested_splits or invalid_splits:
        raise ValueError(
            f'Invalid --splits={args.splits!r}; invalid={invalid_splits}'
        )

    logger = _setup_logger()
    exp_dir = _resolve_exp_dir(args.exp_dir)
    best_pth = os.path.join(exp_dir, 'best.pth')
    config_pkl = os.path.join(exp_dir, 'config.pkl')

    # Argv hack: run.py imports trigger argparse, so strip our flags
    sys_argv_save = sys.argv
    sys.argv = sys.argv[:1]
    from utils import datasets_utils
    from run import trainer
    sys.argv = sys_argv_save

    with open(config_pkl, 'rb') as f:
        run_args = pickle.load(f)

    # A completed student may itself have been trained with KD.  Its teacher
    # cache is not an inference dependency and must not be loaded again while
    # exporting this student's own predictions (especially in a
    # Validation-only protocol).
    run_args.teacher_predictions_path = 'none'
    run_args.kd_alpha = 0.0
    run_args.teacher_predictions_strict_alignment = 0

    if args.retrieval_cache_override is not None:
        override_path = os.path.abspath(args.retrieval_cache_override)
        if not os.path.isfile(override_path):
            raise FileNotFoundError(
                f'retrieval-cache override not found: {override_path}'
            )
        run_args.analog_retrieval_cache = override_path

    run_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_args.cache_data_path = 'datasets'
    run_args.prob_factor = 5
    if isinstance(getattr(run_args, 'k_hops', '1,2'), str):
        run_args.k_hops = [int(x) for x in run_args.k_hops.split(',')]

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
    dataloader, num_nodes, scaler, num_test_samples = datasets_utils.load_dataset(
        run_args, logger
    )
    run_args.num_nodes = num_nodes

    engine = trainer(
        scaler=scaler, device=device, args=run_args, inverse=False
    )
    engine.model.load_state_dict(torch.load(best_pth, map_location=device))
    engine.model.eval()
    logger.info(f'Loaded teacher from {best_pth}')

    sample_counts = {
        k: dataloader[f'{k}_loader'].size
        for k in requested_splits
        if k != 'test'
    }
    # The test loader pads to a full batch. The cache must retain only the
    # official split rows; DataLoader_time will reproduce padding when needed.
    if 'test' in requested_splits:
        sample_counts['test'] = int(num_test_samples)
    logger.info(f'Sample counts: {sample_counts}')

    output = {}
    for split in requested_splits:
        logger.info(f'Forward {split} ...')
        output[f'teacher_pred_{split}'] = _forward_split(
            engine, dataloader, split, sample_counts[split], device
        )
        logger.info(
            f'{split} pred shape={output[f"teacher_pred_{split}"].shape}'
        )

    output['scaler_mean'] = scaler.mean if hasattr(scaler, 'mean') else np.zeros(1)
    output['scaler_std'] = scaler.std if hasattr(scaler, 'std') else np.ones(1)
    output['prediction_space'] = np.asarray('normalized')
    output['teacher_name'] = np.asarray(
        args.teacher_name or Path(exp_dir).name
    )
    output['dataset_name'] = np.asarray(run_args.dataset_name)
    output['input_length'] = np.asarray(run_args.input_length, dtype=np.int64)
    output['predict_length'] = np.asarray(
        run_args.predict_length, dtype=np.int64
    )
    output['num_nodes'] = np.asarray(num_nodes, dtype=np.int64)
    output['source_checkpoint'] = np.asarray(str(Path(best_pth).resolve()))

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path + '.tmp.npz'
    np.savez_compressed(tmp, **output)
    os.replace(tmp, output_path)
    logger.info(f'Saved teacher predictions to {output_path}')


if __name__ == '__main__':
    main()
