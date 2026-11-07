"""Standalone NumPy decoder for the online residual-memory NPZ formats."""
from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


def _delta_codec_module():
    path = Path(__file__).resolve().parent / 'delta_bitpacked_state_codec.py'
    spec = importlib.util.spec_from_file_location('delta_bitpacked_state_codec', path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load delta codec implementation: {path}')
    spec.loader.exec_module(module)
    return module


def decode_online_state(path: str | Path) -> dict[str, np.ndarray]:
    """Decode an archive to the uncompressed OnlineResidualMemory state mapping."""
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    if 'train_raw_delta_codec_id' in state:
        codec_id = int(np.asarray(state.pop('train_raw_delta_codec_id')).item())
        if codec_id != 1:
            raise ValueError(f'Unsupported train_raw delta codec id: {codec_id}')
        shape_values = np.asarray(state.pop('train_raw_delta_shape'), dtype=np.int64)
        if shape_values.shape != (2,):
            raise ValueError('Malformed delta-bitpacked Train history shape')
        codec = _delta_codec_module()
        raw = codec.decode_raw(
            state.pop('train_raw_delta_prefix'),
            state.pop('train_raw_delta_bits'),
            state.pop('train_raw_delta_widths'),
            tuple(int(v) for v in shape_values),
            state.pop('train_raw_delta_sha256'))
        state['train_raw'] = raw.astype(np.float32)
    if 'idx_train_start' in state:
        start = int(np.asarray(state.pop('idx_train_start')).item())
        count = int(np.asarray(state.pop('idx_train_count')).item())
        if count <= 0:
            raise ValueError('Invalid contiguous index count')
        state['idx_train'] = np.arange(start, start + count, dtype=np.int32)
    if 'train_raw_mixed_shape' in state:
        shape = tuple(int(v) for v in state.pop('train_raw_mixed_shape'))
        q_nodes = np.asarray(state.pop('train_raw_mixed_q_nodes'), dtype=np.int64)
        exact_nodes = np.asarray(state.pop('train_raw_mixed_exact_nodes'), dtype=np.int64)
        codes = state.pop('train_raw_mixed_q_codes')
        scale = state.pop('train_raw_mixed_q_scale')
        exact = state.pop('train_raw_mixed_exact')
        tail = state.pop('train_raw_mixed_tail')
        if (len(shape) != 2 or shape[0] < 0 or shape[1] <= 0 or
                len(q_nodes) + len(exact_nodes) != shape[1] or
                not np.array_equal(np.sort(np.r_[q_nodes, exact_nodes]),
                                   np.arange(shape[1])) or
                codes.shape != (shape[0], len(q_nodes)) or
                exact.shape != (shape[0], len(exact_nodes)) or
                scale.shape != (len(q_nodes),) or
                tail.ndim != 2 or tail.shape[1] != shape[1]):
            raise ValueError('Malformed mixed-history state')
        prefix = np.empty(shape, dtype=np.float32)
        prefix[:, q_nodes] = codes.astype(np.float32) * scale[None]
        prefix[:, exact_nodes] = exact.astype(np.float32)
        state['train_raw'] = np.concatenate((prefix, tail.astype(np.float32)), axis=0)
    required = {'train_raw', 'idx_train', 'train_tod', 'node_ids', 'quant_scale'}
    missing = required.difference(state)
    if missing:
        raise ValueError(f'Missing decoded state fields: {sorted(missing)}')
    return state
