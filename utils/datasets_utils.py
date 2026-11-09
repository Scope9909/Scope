import os
import torch
import numpy as np
import threading
import multiprocessing as mp
import pandas as pd
from datetime import datetime

file_root_path = './datasets'
file_path_dict = {
                 'PEMS04':f'{file_root_path}/PEMS04/data.npz',
                 'PEMS08':f'{file_root_path}/PEMS08/data.npz',
                 'PEMS07':f'{file_root_path}/PEMS07/data.npz',
                 'PEMS03':f'{file_root_path}/PEMS03/data.npz',
                 'PEMSBAY':f'{file_root_path}/PEMSBAY/data.npz',
                 'CA':f'{file_root_path}/ca/ca_his_2019.h5',
                 'CHIBike':f'{file_root_path}/CHIBike/CHIBike.grid',
                 'T-Drive':f'{file_root_path}/T-Drive/T-Drive.grid',
                 'NYCTaxi':f'{file_root_path}/NYCTaxi/NYCTaxi.grid'
                }
meta_info_dict= {
        'PEMS03':['2018-09-01 00:00:00','5m'],
        'PEMS04': ['2018-01-01 00:00:00', '5m'],
        'PEMS08': ['2016-07-01 00:00:00', '5m'],
        'PEMS07': ['2017-05-01 00:00:00', '5m'],
        'PEMSBAY': ['2017-01-01 00:00:00', '5m'],
        'CHIBike':['2020-07-01 00:00:00', '30m'],
        'NYCTaxi':['2014-01-01 00:00:00', '30m'],
        'T-Drive':['2015-02-01 00:00:00', '1h'],
        'CA':['2019-01-01 00:00:00', '15m']
    }

class DataLoader_time(object):
    def __init__(self, data, time_data, idx, seq_len, horizon, bs, logger,
                 pad_last_sample=False, scaler=None, drop_last=True,
                 retrieval_data=None, exact_retrieval_data=None,
                 exact_available_data=None, teacher_predictions_data=None,
                 retrieval_quality_data=None,
                 weekly_retrieval_data=None,
                 metric_retrieval_data=None,
                 analog_candidate_index_data=None,
                 analog_candidate_code_data=None,
                 analog_candidate_parent_prediction_data=None,
                 analog_candidate_parent_index_data=None):
        if retrieval_data is not None:
            retrieval_data = np.asarray(retrieval_data, dtype=np.float32)
            if len(retrieval_data) != len(idx):
                raise ValueError(
                    'retrieval_data must have one entry per sample index: '
                    f'{len(retrieval_data)} != {len(idx)}'
                )
        if weekly_retrieval_data is not None:
            weekly_retrieval_data = np.asarray(
                weekly_retrieval_data, dtype=np.float32
            )
            if retrieval_data is None:
                raise ValueError(
                    'weekly_retrieval_data requires retrieval_data'
                )
            if weekly_retrieval_data.shape != retrieval_data.shape:
                raise ValueError(
                    'weekly_retrieval_data must match retrieval_data exactly: '
                    f'{weekly_retrieval_data.shape} vs '
                    f'{retrieval_data.shape}'
                )
            retrieval_data = np.concatenate(
                [retrieval_data, weekly_retrieval_data], axis=-1
            )
        if metric_retrieval_data is not None:
            metric_retrieval_data = np.asarray(
                metric_retrieval_data, dtype=np.float32
            )
            if retrieval_data is None:
                raise ValueError(
                    'metric_retrieval_data requires retrieval_data'
                )
            if metric_retrieval_data.shape != retrieval_data.shape:
                raise ValueError(
                    'metric_retrieval_data must match retrieval_data exactly: '
                    f'{metric_retrieval_data.shape} vs '
                    f'{retrieval_data.shape}'
                )
            retrieval_data = np.concatenate(
                [retrieval_data, metric_retrieval_data], axis=-1
            )
        if retrieval_quality_data is not None:
            retrieval_quality_data = np.asarray(
                retrieval_quality_data, dtype=np.float32
            )
            if retrieval_data is None:
                raise ValueError(
                    'retrieval_quality_data requires retrieval_data'
                )
            # retrieval_data is (N, H, N_nodes, C_retrieval=1) and
            # retrieval_quality_data is (N, H, N_nodes, C_quality>=1).
            # They are concatenated along the last (channel) axis, so the
            # spatial/temporal axes must match; the channel axis may differ.
            if (
                retrieval_quality_data.shape[:-1]
                != retrieval_data.shape[:-1]
            ):
                raise ValueError(
                    'retrieval_quality_data spatial/temporal shape does not '
                    f'match retrieval_data: {retrieval_quality_data.shape} '
                    f'vs {retrieval_data.shape}'
                )
            retrieval_data = np.concatenate(
                [retrieval_data, retrieval_quality_data], axis=-1
            )
        if (exact_retrieval_data is None) != (exact_available_data is None):
            raise ValueError(
                'exact_retrieval_data and exact_available_data must be '
                'provided together'
            )
        if exact_retrieval_data is not None:
            # Preserve the cache's own dtype (e.g. float16 for fp16 caches)
            # to halve on-GPU memory. Adapter modules cast to float32
            # internally when needed.
            exact_retrieval_data = np.asarray(exact_retrieval_data)
            if not np.issubdtype(exact_retrieval_data.dtype, np.floating):
                exact_retrieval_data = exact_retrieval_data.astype(
                    np.float32
                )
            exact_available_data = np.asarray(
                exact_available_data, dtype=np.uint8
            )
            if not (
                len(exact_retrieval_data) == len(idx)
                and len(exact_available_data) == len(idx)
            ):
                raise ValueError(
                    'Exact retrieval arrays must have one entry per sample'
                )
            if exact_retrieval_data.shape != exact_available_data.shape:
                raise ValueError(
                    'Exact retrieval and availability shapes must match'
                )
        if teacher_predictions_data is not None:
            teacher_predictions_data = np.asarray(
                teacher_predictions_data, dtype=np.float32
            )
            if len(teacher_predictions_data) != len(idx):
                raise ValueError(
                    'teacher_predictions_data must have one entry per '
                    f'sample: {len(teacher_predictions_data)} != {len(idx)}'
                )
        if analog_candidate_index_data is not None:
            analog_candidate_index_data = np.asarray(
                analog_candidate_index_data, dtype=np.int32
            )
            if analog_candidate_index_data.ndim != 3:
                raise ValueError(
                    'analog_candidate_index_data must be shaped '
                    '(sample,K,node)'
                )
            if (
                len(analog_candidate_index_data) != len(idx)
                or analog_candidate_index_data.shape[2] != data.shape[1]
            ):
                raise ValueError(
                    'Candidate index cache must have one (K,node) entry per '
                    'sample and match the dataset node count'
                )
            valid_endpoint = analog_candidate_index_data >= 0
            if np.any(
                analog_candidate_index_data[valid_endpoint]
                < int(seq_len) - 1
            ):
                raise ValueError(
                    'Candidate endpoint does not contain a full input history'
                )
            if np.any(
                analog_candidate_index_data[valid_endpoint]
                + int(horizon)
                >= len(data)
            ):
                raise ValueError(
                    'Candidate endpoint plus prediction horizon exceeds data'
                )
        if analog_candidate_code_data is not None:
            analog_candidate_code_data = np.asarray(
                analog_candidate_code_data
            )
            if analog_candidate_index_data is None:
                raise ValueError(
                    'analog_candidate_code_data requires candidate indices'
                )
            if analog_candidate_code_data.ndim != 5:
                raise ValueError(
                    'analog_candidate_code_data must be shaped '
                    '(sample,K,rank,node,channel)'
                )
            if (
                len(analog_candidate_code_data) != len(idx)
                or analog_candidate_code_data.shape[1] != analog_candidate_index_data.shape[1]
                or analog_candidate_code_data.shape[3] != data.shape[1]
                or analog_candidate_code_data.shape[4] != data.shape[2]
            ):
                raise ValueError(
                    'Candidate code cache must align with sample/K/node/channel '
                    f'axes, got {analog_candidate_code_data.shape}'
                )
            if not np.issubdtype(analog_candidate_code_data.dtype, np.floating):
                analog_candidate_code_data = analog_candidate_code_data.astype(
                    np.float32
                )
            if not np.isfinite(analog_candidate_code_data).all():
                raise ValueError('Candidate code cache contains non-finite values')
        if (
            analog_candidate_parent_prediction_data is None
        ) != (analog_candidate_parent_index_data is None):
            raise ValueError(
                'Candidate parent predictions and their train indices must '
                'be provided together'
            )
        candidate_parent_position = None
        if analog_candidate_parent_prediction_data is not None:
            if analog_candidate_code_data is not None:
                raise ValueError(
                    'Candidate-parent residual features currently require '
                    'full candidate futures, not PCA candidate codes'
                )
            analog_candidate_parent_prediction_data = np.asarray(
                analog_candidate_parent_prediction_data, dtype=np.float32
            )
            analog_candidate_parent_index_data = np.asarray(
                analog_candidate_parent_index_data, dtype=np.int64
            )
            expected_parent_tail = (
                int(horizon), data.shape[1], data.shape[2]
            )
            if (
                analog_candidate_parent_prediction_data.ndim != 4
                or analog_candidate_parent_prediction_data.shape[1:]
                != expected_parent_tail
            ):
                raise ValueError(
                    'Candidate parent predictions must be shaped '
                    f'(train_sample,{expected_parent_tail}); got '
                    f'{analog_candidate_parent_prediction_data.shape}'
                )
            if (
                analog_candidate_parent_index_data.ndim != 1
                or len(analog_candidate_parent_index_data)
                != len(analog_candidate_parent_prediction_data)
            ):
                raise ValueError(
                    'Candidate parent train indices must be one-dimensional '
                    'and align with the parent predictions'
                )
            if (
                np.any(analog_candidate_parent_index_data < 0)
                or np.any(analog_candidate_parent_index_data >= len(data))
                or len(np.unique(analog_candidate_parent_index_data))
                != len(analog_candidate_parent_index_data)
            ):
                raise ValueError(
                    'Candidate parent train indices must be unique valid '
                    'global endpoints'
                )
            if not np.isfinite(
                analog_candidate_parent_prediction_data
            ).all():
                raise ValueError(
                    'Candidate parent prediction cache contains non-finite values'
                )
            candidate_parent_position = np.full(
                len(data), -1, dtype=np.int64
            )
            candidate_parent_position[
                analog_candidate_parent_index_data
            ] = np.arange(
                len(analog_candidate_parent_index_data), dtype=np.int64
            )
        # Preserve each sample's chronological position even after training
        # shuffles. This is derived from the current split only.
        idx_min = int(np.min(idx))
        idx_max = int(np.max(idx))
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
            if retrieval_data is not None:
                retrieval_padding = np.repeat(
                    retrieval_data[-1:], num_padding, axis=0
                )
                retrieval_data = np.concatenate(
                    [retrieval_data, retrieval_padding], axis=0
                )
            if exact_retrieval_data is not None:
                exact_retrieval_padding = np.repeat(
                    exact_retrieval_data[-1:], num_padding, axis=0
                )
                exact_available_padding = np.repeat(
                    exact_available_data[-1:], num_padding, axis=0
                )
                exact_retrieval_data = np.concatenate(
                    [exact_retrieval_data, exact_retrieval_padding], axis=0
                )
                exact_available_data = np.concatenate(
                    [exact_available_data, exact_available_padding], axis=0
                )
            if teacher_predictions_data is not None:
                teacher_padding = np.repeat(
                    teacher_predictions_data[-1:], num_padding, axis=0
                )
                teacher_predictions_data = np.concatenate(
                    [teacher_predictions_data, teacher_padding], axis=0
                )
            if analog_candidate_index_data is not None:
                candidate_padding = np.repeat(
                    analog_candidate_index_data[-1:], num_padding, axis=0
                )
                analog_candidate_index_data = np.concatenate(
                    [analog_candidate_index_data, candidate_padding], axis=0
                )
            if analog_candidate_code_data is not None:
                code_padding = np.repeat(
                    analog_candidate_code_data[-1:], num_padding, axis=0
                )
                analog_candidate_code_data = np.concatenate(
                    [analog_candidate_code_data, code_padding], axis=0
                )
        
        self.data = data
        self.time_data = time_data
        self.idx = idx
        idx_span = max(idx_max - idx_min, 1)
        self.temporal_position = (
            (idx.astype(np.float32) - float(idx_min)) / float(idx_span)
        ).astype(np.float32, copy=False)
        self.retrieval_data = retrieval_data
        self.exact_retrieval_data = exact_retrieval_data
        self.exact_available_data = exact_available_data
        self.teacher_predictions_data = teacher_predictions_data
        self.analog_candidate_index_data = analog_candidate_index_data
        self.analog_candidate_code_data = analog_candidate_code_data
        self.analog_candidate_parent_prediction_data = (
            analog_candidate_parent_prediction_data
        )
        self.analog_candidate_parent_position = candidate_parent_position
        # Read by the KD-aware train loop *after* each get_iterator() step;
        # None when the loader was not configured with teacher soft targets.
        self.current_teacher_batch = None
        self.current_temporal_position_batch = None
        self.size = len(idx)
        self.bs = bs
        self.drop_last = bool(drop_last)
        if self.drop_last:
            self.num_batch = int(self.size // self.bs)
        else:
            self.num_batch = int(np.ceil(self.size / self.bs))
        self.scaler = scaler
        self.current_ind = 0
        logger.info(
            'Sample num: ' + str(self.idx.shape[0])
            + ', Batch num: ' + str(self.num_batch)
            + ', Drop last: ' + str(self.drop_last)
        )
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx
        self.temporal_position = self.temporal_position[perm]
        if self.retrieval_data is not None:
            self.retrieval_data = self.retrieval_data[perm]
        if self.exact_retrieval_data is not None:
            self.exact_retrieval_data = self.exact_retrieval_data[perm]
            self.exact_available_data = self.exact_available_data[perm]
        if self.teacher_predictions_data is not None:
            # Teacher predictions are stored one-to-one with idx. They must
            # follow the same permutation or KD pairs each query with another
            # sample's forecast after the first training shuffle.
            self.teacher_predictions_data = self.teacher_predictions_data[perm]
        if self.analog_candidate_index_data is not None:
            self.analog_candidate_index_data = (
                self.analog_candidate_index_data[perm]
            )
        if self.analog_candidate_code_data is not None:
            self.analog_candidate_code_data = self.analog_candidate_code_data[perm]


    def write_to_shared_array(self, x, y, x_time_data, y_time_data, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            if self.scaler is not None:
                x[i]= self.scaler.transform(x[i])
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :]
            x_time_data[i] = self.time_data[idx_ind[i] + self.x_offsets, :]
            y_time_data[i] = self.time_data[idx_ind[i] + self.y_offsets, :]

    def _build_analog_candidate_batch(
        self, idx_ind, candidate_endpoint, candidate_code=None
    ):
        """Build causal candidate futures and history-only ranking features."""
        candidate_endpoint = np.asarray(candidate_endpoint, dtype=np.int64)
        batch, candidates, nodes = candidate_endpoint.shape
        if nodes != self.data.shape[1]:
            raise ValueError('Candidate node axis does not match raw data')
        endpoint_valid = candidate_endpoint >= 0
        safe_endpoint = np.where(
            endpoint_valid, candidate_endpoint, self.seq_len - 1
        )
        node_index = np.arange(nodes, dtype=np.int64)[
            None, None, :, None
        ]

        future_time = (
            safe_endpoint[..., None]
            + self.y_offsets[None, None, None, :]
        )
        candidate_future_raw = self.data[
            future_time, node_index, 0
        ].astype(np.float32, copy=False)
        candidate_available = (
            endpoint_valid[..., None] & (candidate_future_raw != 0)
        )

        scaler_mean = np.asarray(self.scaler.mean, dtype=np.float32).reshape(-1)
        scaler_std = np.asarray(self.scaler.std, dtype=np.float32).reshape(-1)
        if np.any(scaler_std <= 0):
            raise ValueError('Dataset scaler std must be positive')
        if scaler_std.size == 1:
            if candidate_code is None:
                candidate_future = (
                    candidate_future_raw - float(scaler_mean[0])
                ) / float(scaler_std[0])
            node_std = np.full(
                (1, 1, 1, nodes), float(scaler_std[0]), dtype=np.float32
            )
        elif scaler_std.size == nodes:
            if candidate_code is None:
                candidate_future = (
                    candidate_future_raw
                    - scaler_mean[None, None, :, None]
                ) / scaler_std[None, None, :, None]
            node_std = scaler_std[None, None, None, :]
        else:
            raise ValueError(
                'Scaler must be global or contain one statistic per node'
            )
        if candidate_code is None:
            candidate_future = candidate_future.transpose(0, 1, 3, 2)[
                ..., None
            ].astype(np.float32, copy=False)
        candidate_available = candidate_available.transpose(
            0, 1, 3, 2
        )[..., None].astype(np.uint8, copy=False)

        if self.analog_candidate_parent_prediction_data is not None:
            position = self.analog_candidate_parent_position[safe_endpoint]
            missing_parent = endpoint_valid & (position < 0)
            if np.any(missing_parent):
                raise ValueError(
                    'A valid candidate endpoint is absent from the '
                    'train-only parent prediction cache'
                )
            safe_position = np.where(position >= 0, position, 0)
            node_lookup = np.arange(nodes, dtype=np.int64)[None, None, :]
            candidate_parent = self.analog_candidate_parent_prediction_data[
                safe_position, :, node_lookup, :
            ]
            # Advanced indexing yields (B,K,N,H,C); align it with the
            # candidate future payload (B,K,H,N,C).
            candidate_parent = candidate_parent.transpose(0, 1, 3, 2, 4)
            candidate_parent = np.where(
                endpoint_valid[:, :, None, :, None],
                candidate_parent,
                0.0,
            ).astype(np.float32, copy=False)
            candidate_future = np.concatenate(
                [candidate_future, candidate_parent], axis=-1
            )

        query_history = self.data[
            idx_ind[:, None] + self.x_offsets[None, :], :, 0
        ].astype(np.float32, copy=False)
        candidate_history_time = (
            safe_endpoint[..., None]
            + self.x_offsets[None, None, None, :]
        )
        candidate_history = self.data[
            candidate_history_time, node_index, 0
        ].transpose(0, 1, 3, 2).astype(np.float32, copy=False)
        query_history = query_history[:, None, :, :]
        valid_history = (
            endpoint_valid[:, :, None, :]
            & (query_history != 0)
            & (candidate_history != 0)
        )
        valid_float = valid_history.astype(np.float32)
        valid_count = valid_float.sum(axis=2)
        safe_count = np.maximum(valid_count, 1.0)
        difference_z = (
            query_history - candidate_history
        ) / node_std
        difference_z = np.where(valid_history, difference_z, 0.0)

        mse_z = (difference_z ** 2).sum(axis=2) / safe_count
        mae_z = np.abs(difference_z).sum(axis=2) / safe_count
        mean_difference_z = difference_z.sum(axis=2) / safe_count
        last_difference_z = np.where(
            valid_history[:, :, -1, :],
            difference_z[:, :, -1, :],
            0.0,
        )

        edge_valid = (
            endpoint_valid
            & (query_history[:, :, 0, :] != 0)
            & (query_history[:, :, -1, :] != 0)
            & (candidate_history[:, :, 0, :] != 0)
            & (candidate_history[:, :, -1, :] != 0)
        )
        query_slope = (
            query_history[:, :, -1, :] - query_history[:, :, 0, :]
        ) / node_std[:, :, 0, :]
        candidate_slope = (
            candidate_history[:, :, -1, :]
            - candidate_history[:, :, 0, :]
        ) / node_std[:, :, 0, :]
        slope_difference_z = np.where(
            edge_valid, query_slope - candidate_slope, 0.0
        )

        consecutive_valid = (
            valid_history[:, :, 1:, :] & valid_history[:, :, :-1, :]
        )
        consecutive_float = consecutive_valid.astype(np.float32)
        consecutive_count = np.maximum(
            consecutive_float.sum(axis=2), 1.0
        )
        query_delta = np.abs(
            query_history[:, :, 1:, :] - query_history[:, :, :-1, :]
        ) / node_std
        candidate_delta = np.abs(
            candidate_history[:, :, 1:, :]
            - candidate_history[:, :, :-1, :]
        ) / node_std
        volatility_difference_z = (
            np.where(consecutive_valid, query_delta, 0.0).sum(axis=2)
            - np.where(consecutive_valid, candidate_delta, 0.0).sum(axis=2)
        ) / consecutive_count

        if candidates > 1:
            candidate_rank = (
                np.arange(candidates, dtype=np.float32) / (candidates - 1)
            )
        else:
            candidate_rank = np.zeros(1, dtype=np.float32)
        candidate_rank = np.broadcast_to(
            candidate_rank[None, :, None], (batch, candidates, nodes)
        )
        candidate_features = np.stack([
            np.log1p(mse_z),
            np.log1p(mae_z),
            np.tanh(last_difference_z / 3.0),
            np.tanh(mean_difference_z / 3.0),
            np.tanh(slope_difference_z / 3.0),
            np.tanh(volatility_difference_z / 3.0),
            valid_count / float(self.seq_len),
            candidate_rank,
        ], axis=-1).astype(np.float32, copy=False)
        candidate_features = np.clip(candidate_features, -8.0, 8.0)
        candidate_payload = (
            candidate_code if candidate_code is not None else candidate_future
        )
        if not (
            np.isfinite(candidate_payload).all()
            and np.isfinite(candidate_features).all()
        ):
            raise ValueError('Non-finite analog candidate batch')
        return (
            candidate_payload,
            candidate_available,
            candidate_features,
        )

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)
                
                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], self.data.shape[-1])
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                x_time_data_shape = (len(idx_ind), self.seq_len, 2)
                x_time_data_shared = mp.RawArray('i', int(np.prod(x_time_data_shape)))
                x_time_data = np.frombuffer(x_time_data_shared, dtype='i').reshape(x_time_data_shape)

                y_time_data_shape = (len(idx_ind), self.horizon, 2)
                y_time_data_shared = mp.RawArray('i', int(np.prod(y_time_data_shape)))
                y_time_data = np.frombuffer(y_time_data_shared, dtype='i').reshape(y_time_data_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 1
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, x_time_data, y_time_data, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                model_inputs = [x, x_time_data]
                if self.retrieval_data is not None:
                    model_inputs.append(
                        self.retrieval_data[start_ind:end_ind]
                    )
                if self.exact_retrieval_data is not None:
                    model_inputs.extend([
                        self.exact_retrieval_data[start_ind:end_ind],
                        self.exact_available_data[start_ind:end_ind],
                    ])
                if self.analog_candidate_index_data is not None:
                    candidate_batch = self._build_analog_candidate_batch(
                        idx_ind,
                        self.analog_candidate_index_data[start_ind:end_ind],
                        None if self.analog_candidate_code_data is None else
                        self.analog_candidate_code_data[start_ind:end_ind],
                    )
                    model_inputs.extend(candidate_batch)
                # KD soft targets are NOT folded into model_inputs — that
                # would break unpack_dataloader_input's 2/3/5/8 contract. The
                # training loop reads them via self.current_teacher_batch.
                if self.teacher_predictions_data is not None:
                    self.current_teacher_batch = (
                        self.teacher_predictions_data[start_ind:end_ind]
                    )
                else:
                    self.current_teacher_batch = None
                self.current_temporal_position_batch = (
                    self.temporal_position[start_ind:end_ind]
                )
                yield (model_inputs, [y, y_time_data])
                self.current_ind += 1

        return _wrapper()



class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _broadcast(self, arr, data):
        """把 mean/std 重整为能在「节点轴」上广播的形状，且返回类型与 data 一致。
        - 标量(global)：原样返回(numpy 路径)；torch 路径下包成 data 所在设备的 0 维张量，
          避免 CUDA 张量与 numpy 标量直接运算触发 numpy 转换报错。
        - 数组(node-wise, 形状 (N,) 或 (N,1))：节点轴通过匹配 data 中大小等于 N 的维度动态定位，
          因此无论 data 是 (T,N,C) / (B,T,N,C) / (S,N) 都能正确对齐。"""
        if isinstance(data, np.ndarray):
            if np.ndim(arr) == 0:
                return arr
            N = arr.shape[0]
            node_axes = [i for i, s in enumerate(data.shape) if s == N]
            node_axis = node_axes[0] if node_axes else (data.ndim - 1)
            shape = [1] * data.ndim
            shape[node_axis] = N
            return np.reshape(arr, shape)
        else:  # torch.Tensor：始终返回 data 所在设备的张量（标量也转，规避 CUDA+numpy 标量不兼容）
            t = torch.as_tensor(arr, device=data.device, dtype=data.dtype)
            if t.ndim == 0:
                return t
            N = t.shape[0]
            node_axes = [i for i, s in enumerate(data.shape) if s == N]
            node_axis = node_axes[0] if node_axes else (data.ndim - 1)
            shape = [1] * data.ndim
            shape[node_axis] = N
            return t.reshape(shape)

    def transform(self, data):
        if isinstance(data, np.ndarray):
            mean = self._broadcast(self.mean, data)
            std = self._broadcast(self.std, data)
            return (data - mean) / std
        elif isinstance(data, torch.Tensor):
            mean = self._broadcast(self.mean, data)
            std = self._broadcast(self.std, data)
            return (data - mean) / std

    def inverse_transform(self, data):
        if isinstance(data, np.ndarray):
            mean = self._broadcast(self.mean, data)
            std = self._broadcast(self.std, data)
            return (data * std) + mean
        elif isinstance(data, torch.Tensor):
            mean = self._broadcast(self.mean, data)
            std = self._broadcast(self.std, data)
            return (data * std) + mean


def load_grid(filename):
    gridfile = pd.read_csv(filename)
    df = gridfile[['inflow', 'outflow']]
    max_row_id = gridfile['row_id'].max()
    max_col_id = gridfile['column_id'].max()
    num_nodes = (max_row_id+1)*(max_col_id+1)

    timesolts = list(gridfile['time'][:int(gridfile.shape[0] / num_nodes)])
    len_time = len(timesolts)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i + len_time].values)
    data = np.array(data, dtype=float)
    return data.swapaxes(0, 1)

def load_dyna(filename):
    dynafile = pd.read_csv(filename)
    df = dynafile[['traffic_flow']]
    num_nodes = dynafile['entity_id'].max()+1

    timesolts = list(dynafile['time'][:int(dynafile.shape[0] / num_nodes)])
    len_time = len(timesolts)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i+len_time].values)
    data = np.array(data, dtype=np.float)
    return data.swapaxes(0, 1)

def load_raw_data(raw_data_path):
    df_time = None
    # load raw data
    if raw_data_path.endswith('npz'):
        data = np.load(raw_data_path)['data']
    elif raw_data_path.endswith('grid'):
        data = load_grid(raw_data_path)
    elif raw_data_path.endswith('dyna'):
        data = load_dyna(raw_data_path)
    elif raw_data_path.endswith('h5'):
        ori = pd.read_hdf(raw_data_path)
        data = ori.values #(length, N)
        data = np.expand_dims(data, axis=-1)
        df_time = pd.DataFrame({'date':ori.index})
    return data, df_time

def generate_datetime_series(start_datetime_str, interval_str, n, col_name='date'):
    start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
    interval = pd.to_timedelta(interval_str)
    date_time_series = [start_datetime + i * interval for i in range(n)]
    df = pd.DataFrame({col_name: date_time_series})
    return df

def generate_time_features(df_time, freq):
    # tod (time of day)
    time_ind = (df_time['date'].values - df_time['date'].values.astype('datetime64[D]')) / pd.to_timedelta(freq)
    t_of_d = time_ind.reshape([time_ind.shape[0],1])
    
    # dow (day of week)
    dow = df_time.date.apply(lambda row:row.dayofweek,1)
    d_of_w = dow.values.reshape([dow.shape[0],1])
    
    return np.concatenate([t_of_d, d_of_w], axis=-1).astype(np.int64)

def create_dataset_files(args, data, df_time):
    output_dir = f'{args.cache_data_path}/{args.dataset_name}'
    dataset_name = args.dataset_name
    input_length = args.input_length
    predict_length = args.predict_length
    train_ratio = args.train_ratio
    test_ratio = args.test_ratio

    os.makedirs(output_dir, exist_ok=True)
    start_time_str, freq = meta_info_dict[dataset_name]
    if df_time is None:        
        df_time = generate_datetime_series(start_time_str, freq, data.shape[0], 'date')

    print(f"data.shape: {data.shape}")
    print(f"df_time.shape:{df_time.shape}")
    
    # generate time idx features (tod and dow)
    time_data = generate_time_features(df_time, freq)
    print(f"time_data.shape: {time_data.shape}")
    
    x_offsets = np.arange(-(input_length - 1), 1, 1)
    y_offsets = np.arange(1, (predict_length + 1), 1)
    print('x_offsets:',x_offsets)
    print('y_offsets:',y_offsets)

    total_length, num_nodes, _ = data.shape
    min_t = abs(min(x_offsets))
    max_t = abs(total_length - abs(max(y_offsets)))
    idx = np.arange(min_t, max_t, 1)

    num_samples = len(idx)
    
    split_line1 = round(num_samples * train_ratio)
    split_line2 = round(num_samples * (1 - test_ratio))
    
    idx_train = idx[:split_line1]
    idx_val = idx[split_line1:split_line2]
    idx_test = idx[split_line2:]
    
    nodewise = bool(getattr(args, 'nodewise_scaler', 0))
    x_train = data[:idx_val[0] - input_length, :, 0]   # (T_tr, N)
    if nodewise:
        # 按节点(node-wise)分别求均值/标准差 —— 标准交通预测论文惯例(DCRNN/STGCN/AGCRN等)
        # 形状保持 (N,1)，使其能对 (T,N,1) 数据在「节点轴」正确广播
        x_train_mean = x_train.mean(axis=0, keepdims=True).reshape(-1, 1)   # (N,1)
        x_train_std = x_train.std(axis=0, keepdims=True).reshape(-1, 1)     # (N,1)
        scaler_tag = 'index_scaler_nodewise'
    else:
        # 全局(global)均值/标准差 —— 当前默认行为
        x_train_mean = x_train.mean()                  # scalar
        x_train_std = x_train.std()                    # scalar
        scaler_tag = 'index_scaler'

    with open(os.path.join(output_dir, f'{dataset_name}_{input_length}_{predict_length}_description.txt'),mode='w')as f:
        f.write('data.shape='+str(data.shape)+'\n')
        f.write('time_data.shape='+str(time_data.shape)+'\n')
        # f.write('data_tod[90:102]:'+str(time_data[90:102,0])+'\n')
        # f.write('data_dow[90:102]:'+str(time_data[90:102,1])+'\n\n')
        f.write('idx.shape='+str(idx.shape)+'\n')
        f.write('idx_train.shape='+str(idx_train.shape)+'\n')
        f.write('idx_val.shape='+str(idx_val.shape)+'\n')
        f.write('idx_test.shape='+str(idx_test.shape)+'\n\n')
        f.write('idx:'+str(split_line1)+' '+str(idx_val[0] - input_length)+'\n')
        f.write('idx_train:'+str(idx_train[0])+' '+str(idx_train[-1])+'\n')
        f.write('idx_val:'+str(idx_val[0])+' '+str(idx_val[-1])+'\n')
        f.write('idx_test:'+str(idx_test[0])+' '+str(idx_test[-1])+'\n')
        f.write('scaler_mode='+('nodewise' if nodewise else 'global')+'\n')
        f.write('mean='+str(x_train_mean)+', std='+str(x_train_std)+'\n')

    print('description file has been generated!')

    # np.savez_compressed(f'{output_dir}/{dataset_name}_data.npz', data=data)
    np.savez_compressed(f'{output_dir}/{dataset_name}_time.npz', time=time_data)
    np.savez_compressed(f'{output_dir}/{dataset_name}_{input_length}to{predict_length}_{scaler_tag}.npz',
                       idx_train = idx_train,
                       idx_val = idx_val,
                       idx_test = idx_test,
                       scaler_mean = x_train_mean,
                       scaler_std = x_train_std)
    
    print("Data files has been generated:")
    print(f"- {output_dir}/{dataset_name}_time.npz: {time_data.shape}")
    print(f"- {output_dir}/{dataset_name}_{input_length}to{predict_length}_{scaler_tag}.npz: train{len(idx_train)}, valid{len(idx_val)}, test{len(idx_test)}, scaler_mode={'nodewise' if nodewise else 'global'}")
    return time_data, idx_train, idx_val, idx_test, x_train_mean, x_train_std



def load_dataset(args, logger):
    raw_data_path = file_path_dict[args.dataset_name]
    time_path = f"datasets/{args.dataset_name}/{args.dataset_name}_time.npz"
    _scaler_tag = 'index_scaler_nodewise' if bool(getattr(args, 'nodewise_scaler', 0)) else 'index_scaler'
    _index_override = getattr(args, 'index_scaler_path', 'none')
    if _index_override not in (None, '', 'none'):
        index_path = os.path.normpath(_index_override)
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f'Explicit index/scaler cache not found: {index_path}'
            )
    else:
        index_path = f"datasets/{args.dataset_name}/{args.dataset_name}_{args.input_length}to{args.predict_length}_{_scaler_tag}.npz"

    logger.info(f'loading raw data of {args.dataset_name}...')
    data, df_time = load_raw_data(raw_data_path)
    data = data[...,:args.in_dim]
    logger.info(f'Data shape:{data.shape}')
    
    if not (os.path.exists(time_path) and os.path.exists(index_path)):
        logger.info('creating cache files....')
        time_data, idx_train, idx_val, idx_test, scaler_mean, scaler_std = create_dataset_files(args, data, df_time)
    else:
        logger.info(f'loading cache files from index/scaler: {index_path}')
        time_data = np.load(time_path)['time']
        with np.load(index_path, allow_pickle=False) as index_scaler:
            idx_train = np.array(index_scaler['idx_train'], copy=True)
            idx_val = np.array(index_scaler['idx_val'], copy=True)
            idx_test = np.array(index_scaler['idx_test'], copy=True)
            scaler_mean = np.array(
                index_scaler['scaler_mean'], copy=True
            )
            scaler_std = np.array(
                index_scaler['scaler_std'], copy=True
            )
            if 'train_ratio' in index_scaler:
                cached_train_ratio = float(np.asarray(
                    index_scaler['train_ratio']
                ).item())
                if not np.isclose(
                    cached_train_ratio, float(args.train_ratio),
                    rtol=0.0, atol=1e-12,
                ):
                    raise ValueError(
                        f'Index cache train_ratio={cached_train_ratio} '
                        f'does not match config {args.train_ratio}'
                    )
            if 'test_ratio' in index_scaler:
                cached_test_ratio = float(np.asarray(
                    index_scaler['test_ratio']
                ).item())
                if not np.isclose(
                    cached_test_ratio, float(args.test_ratio),
                    rtol=0.0, atol=1e-12,
                ):
                    raise ValueError(
                        f'Index cache test_ratio={cached_test_ratio} '
                        f'does not match config {args.test_ratio}'
                    )

    logger.info(f'Time Data shape: {time_data.shape}')
    num_nodes = data.shape[1]

    scaler = StandardScaler(mean=scaler_mean, std=scaler_std)

    retrieval_by_split = {'train': None, 'val': None, 'test': None}
    weekly_retrieval_by_split = {
        'train': None, 'val': None, 'test': None
    }
    metric_retrieval_by_split = {
        'train': None, 'val': None, 'test': None
    }
    retrieval_quality_by_split = {'train': None, 'val': None, 'test': None}
    exact_retrieval_by_split = {'train': None, 'val': None, 'test': None}
    exact_available_by_split = {'train': None, 'val': None, 'test': None}
    candidate_index_by_split = {'train': None, 'val': None, 'test': None}
    candidate_code_by_split = {'train': None, 'val': None, 'test': None}
    candidate_parent_prediction_train = None
    use_code_domain = bool(getattr(
        args, 'use_analog_code_domain_aggregation', 0
    ))
    use_exact_match = bool(getattr(
        args, 'use_analog_exact_match_residual', 0
    ))
    use_confidence_retrieval = bool(getattr(
        args, 'use_analog_confidence_aware_retrieval_residual', 0
    ))
    use_weekly_retrieval = bool(getattr(
        args, 'use_analog_weekly_retrieval_residual', 0
    ))
    use_metric_retrieval = bool(getattr(
        args, 'use_analog_metric_retrieval_residual', 0
    ))
    use_candidate_reranker = bool(getattr(
        args, 'use_analog_candidate_reranker', 0
    ))
    if use_code_domain and not use_candidate_reranker:
        raise ValueError(
            'use_analog_code_domain_aggregation requires '
            'use_analog_candidate_reranker=1'
        )
    if sum([
        use_confidence_retrieval,
        use_weekly_retrieval,
        use_metric_retrieval,
    ]) > 1:
        raise ValueError(
            'Confidence-aware, weekly, and metric retrieval use separate '
            'single-variable auxiliary channel layouts'
        )
    if (
        bool(getattr(args, 'use_analog_retrieval_residual', 0))
        or bool(getattr(
            args, 'use_analog_lowrank_retrieval_residual', 0
        ))
        or bool(getattr(
            args, 'use_analog_node_horizon_retrieval_residual', 0
        ))
        or use_confidence_retrieval
        or use_metric_retrieval
        or use_exact_match
    ):
        cache_path = getattr(args, 'analog_retrieval_cache', 'none')
        if cache_path in (None, '', 'none') or not os.path.exists(cache_path):
            raise FileNotFoundError(
                f'Analog retrieval cache not found: {cache_path}'
            )
        expected_indices = {
            'train': idx_train,
            'val': idx_val,
            'test': idx_test,
        }
        expected_shape_tail = (
            args.predict_length, num_nodes, args.pre_dim
        )
        with np.load(cache_path, allow_pickle=False) as retrieval_cache:
            metadata_expectations = {
                'k': int(getattr(args, 'analog_retrieval_k', 4)),
                'input_length': int(args.input_length),
                'predict_length': int(args.predict_length),
                'train_leave_one_out': 1,
            }
            for metadata_key, expected_value in metadata_expectations.items():
                if metadata_key not in retrieval_cache:
                    raise KeyError(
                        f'Analog retrieval cache is missing metadata '
                        f'{metadata_key!r}'
                    )
                actual_value = int(np.asarray(
                    retrieval_cache[metadata_key]
                ).item())
                if actual_value != expected_value:
                    raise ValueError(
                        f'Analog retrieval cache {metadata_key}='
                        f'{actual_value}, expected {expected_value}'
                    )
            expected_aggregation = str(getattr(
                args, 'analog_retrieval_aggregation', 'mean'
            ))
            actual_aggregation = (
                str(np.asarray(retrieval_cache['aggregation']).item())
                if 'aggregation' in retrieval_cache
                else 'mean'
            )
            if actual_aggregation != expected_aggregation:
                raise ValueError(
                    'Analog retrieval cache aggregation='
                    f'{actual_aggregation!r}, expected '
                    f'{expected_aggregation!r}'
                )
            if use_exact_match:
                exact_metadata = {
                    'exact_min_valid_history': int(getattr(
                        args, 'analog_exact_min_valid_history', 6
                    )),
                    'exact_train_leave_one_out': 1,
                }
                for metadata_key, expected_value in exact_metadata.items():
                    if metadata_key not in retrieval_cache:
                        raise KeyError(
                            f'Exact-match cache is missing {metadata_key!r}'
                        )
                    actual_value = int(np.asarray(
                        retrieval_cache[metadata_key]
                    ).item())
                    if actual_value != expected_value:
                        raise ValueError(
                            f'Exact-match cache {metadata_key}='
                            f'{actual_value}, expected {expected_value}'
                        )
                if (
                    'exact_match_rule' not in retrieval_cache
                    or str(np.asarray(
                        retrieval_cache['exact_match_rule']
                    ).item()) != 'masked_mse_eq_0'
                ):
                    raise ValueError(
                        'Exact-match cache rule must be masked_mse_eq_0'
                    )
            cached_mean = np.asarray(retrieval_cache['scaler_mean'])
            cached_std = np.asarray(retrieval_cache['scaler_std'])
            if not (
                np.allclose(cached_mean, scaler_mean, rtol=0.0, atol=1e-7)
                and np.allclose(cached_std, scaler_std, rtol=0.0, atol=1e-7)
            ):
                raise ValueError(
                    'Analog retrieval cache scaler does not match the active '
                    'dataset scaler'
                )
            compressed_retrieval = (
                'retrieval_compression' in retrieval_cache
            )
            quantized_retrieval = (
                'retrieval_quantization' in retrieval_cache
            )
            retrieval_quant_scale = None
            if quantized_retrieval:
                quantization = str(np.asarray(
                    retrieval_cache['retrieval_quantization']
                ).item())
                if quantization != 'symmetric_int8_per_horizon_node':
                    raise ValueError(
                        'Unsupported retrieval quantization: '
                        f'{quantization!r}'
                    )
                if str(np.asarray(retrieval_cache[
                    'retrieval_quantization_fit_split'
                ]).item()) != 'train_only':
                    raise ValueError(
                        'Retrieval quantization scale must be fit on '
                        'train_only'
                    )
                retrieval_quant_scale = np.asarray(
                    retrieval_cache['retrieval_quantization_scale'],
                    dtype=np.float32,
                )
                expected_scale_shape = (
                    args.predict_length, num_nodes, args.pre_dim
                )
                if retrieval_quant_scale.shape != expected_scale_shape:
                    raise ValueError(
                        'Retrieval quantization scale has shape '
                        f'{retrieval_quant_scale.shape}, expected '
                        f'{expected_scale_shape}'
                    )
                if not (
                    np.isfinite(retrieval_quant_scale).all()
                    and np.all(retrieval_quant_scale > 0)
                ):
                    raise ValueError(
                        'Retrieval quantization scale must be finite and '
                        'positive'
                    )
            if compressed_retrieval:
                compression = str(np.asarray(
                    retrieval_cache['retrieval_compression']
                ).item())
                if compression != 'per_node_temporal_pca':
                    raise ValueError(
                        'Unsupported retrieval compression: '
                        f'{compression!r}'
                    )
                if str(np.asarray(retrieval_cache[
                    'retrieval_compression_fit_split'
                ]).item()) != 'train_only':
                    raise ValueError(
                        'Compressed retrieval basis must be fit on train_only'
                    )
                compression_rank = int(np.asarray(
                    retrieval_cache['retrieval_compression_rank']
                ).item())
                temporal_mean = np.asarray(
                    retrieval_cache['retrieval_temporal_mean'],
                    dtype=np.float32,
                )
                temporal_basis = np.asarray(
                    retrieval_cache['retrieval_temporal_basis'],
                    dtype=np.float32,
                )
                expected_mean_shape = (
                    args.predict_length, num_nodes, args.pre_dim
                )
                expected_basis_shape = (
                    num_nodes, args.predict_length, compression_rank
                )
                if temporal_mean.shape != expected_mean_shape:
                    raise ValueError(
                        'Compressed retrieval mean has shape '
                        f'{temporal_mean.shape}, expected {expected_mean_shape}'
                    )
                if temporal_basis.shape != expected_basis_shape:
                    raise ValueError(
                        'Compressed retrieval basis has shape '
                        f'{temporal_basis.shape}, expected '
                        f'{expected_basis_shape}'
                    )

            for split_name, split_indices in expected_indices.items():
                idx_key = f'idx_{split_name}'
                value_key = f'retrieval_{split_name}'
                code_key = f'retrieval_code_{split_name}'
                if idx_key not in retrieval_cache or (
                    value_key not in retrieval_cache
                    and code_key not in retrieval_cache
                ):
                    raise KeyError(
                        f'Analog retrieval cache is missing {idx_key!r} or '
                        f'one of {value_key!r}/{code_key!r}'
                    )
                cached_indices = retrieval_cache[idx_key]
                if not np.array_equal(cached_indices, split_indices):
                    raise ValueError(
                        f'Analog retrieval {split_name} indices do not match '
                        'the active dataset split'
                    )
                if value_key in retrieval_cache:
                    retrieval_payload = np.asarray(
                        retrieval_cache[value_key]
                    )
                    if quantized_retrieval:
                        if retrieval_payload.dtype != np.int8:
                            raise ValueError(
                                f'{value_key} must be int8 for '
                                'symmetric_int8_per_horizon_node'
                            )
                        retrieval_values = (
                            retrieval_payload.astype(np.float32)
                            * retrieval_quant_scale[None, ...]
                        )
                    else:
                        retrieval_values = retrieval_payload.astype(
                            np.float32, copy=False
                        )
                else:
                    retrieval_code = np.asarray(
                        retrieval_cache[code_key], dtype=np.float32
                    )
                    expected_code_shape = (
                        len(split_indices), compression_rank,
                        num_nodes, args.pre_dim,
                    )
                    if retrieval_code.shape != expected_code_shape:
                        raise ValueError(
                            f'Expected {code_key} shape '
                            f'{expected_code_shape}, got '
                            f'{retrieval_code.shape}'
                        )
                    if not np.isfinite(retrieval_code).all():
                        raise ValueError(
                            f'{code_key} contains non-finite values'
                        )
                    retrieval_values = temporal_mean[None, ...] + np.einsum(
                        'srnc,nhr->shnc',
                        retrieval_code,
                        temporal_basis,
                        optimize=True,
                    )
                expected_shape = (len(split_indices),) + expected_shape_tail
                if retrieval_values.shape != expected_shape:
                    raise ValueError(
                        f'Expected {value_key} shape {expected_shape}, got '
                        f'{retrieval_values.shape}'
                    )
                if not np.isfinite(retrieval_values).all():
                    raise ValueError(
                        f'{value_key} contains non-finite values'
                    )
                retrieval_by_split[split_name] = retrieval_values
                if use_confidence_retrieval:
                    distance_key = f'history_distance_{split_name}'
                    dispersion_key = f'future_dispersion_{split_name}'
                    if (
                        distance_key not in retrieval_cache
                        or dispersion_key not in retrieval_cache
                    ):
                        raise KeyError(
                            'Confidence-aware analog retrieval requires '
                            f'{distance_key!r} and {dispersion_key!r}'
                        )
                    history_distance = np.asarray(
                        retrieval_cache[distance_key], dtype=np.float32
                    )
                    future_dispersion = np.asarray(
                        retrieval_cache[dispersion_key], dtype=np.float32
                    )
                    expected_distance_shape = (
                        len(split_indices), num_nodes
                    )
                    if history_distance.shape != expected_distance_shape:
                        raise ValueError(
                            f'Expected {distance_key} shape '
                            f'{expected_distance_shape}, got '
                            f'{history_distance.shape}'
                        )
                    if future_dispersion.shape != expected_shape:
                        raise ValueError(
                            f'Expected {dispersion_key} shape '
                            f'{expected_shape}, got '
                            f'{future_dispersion.shape}'
                        )
                    if not (
                        np.isfinite(history_distance).all()
                        and np.isfinite(future_dispersion).all()
                    ):
                        raise ValueError(
                            'Confidence-aware retrieval quality contains '
                            'non-finite values'
                        )
                    # Quality features were written on the raw traffic scale;
                    # convert them into the same z-score space the forecast
                    # channel lives in so the linear confidence gate is well
                    # conditioned.  history_distance is mean squared error in
                    # raw units (per-node MSE averaged over t=12 history
                    # steps) so dividing by std**2 yields a unitless MSE on
                    # z-scored values.  future_dispersion is std of K
                    # candidate futures in raw units so dividing by std gives
                    # the dispersion on z-scored values.
                    std_arr = np.asarray(scaler_std).reshape(-1)
                    if np.any(std_arr <= 0):
                        raise ValueError(
                            'scaler_std must be positive for confidence '
                            'normalization'
                        )
                    if std_arr.size == 1:
                        # Global scalar std (PEMS04 default): single
                        # normalization factor for every node.
                        std_value = float(std_arr[0])
                        history_distance = history_distance / (
                            std_value ** 2
                        )
                        future_dispersion = future_dispersion / std_value
                    elif std_arr.size == num_nodes:
                        # Node-wise std: broadcast across (sample, horizon).
                        std_per_node = std_arr.astype(np.float32)
                        history_distance = history_distance / (
                            std_per_node ** 2
                        )[None, :]
                        future_dispersion = (
                            future_dispersion
                            / std_per_node[None, None, :, None]
                        )
                    else:
                        raise ValueError(
                            'scaler_std size '
                            f'{std_arr.size} does not match num_nodes '
                            f'{num_nodes} or a scalar'
                        )
                    history_distance = history_distance[:, None, :, None]
                    history_distance = np.repeat(
                        history_distance, args.predict_length, axis=1
                    )
                    retrieval_quality_by_split[split_name] = np.concatenate(
                        [history_distance, future_dispersion], axis=-1
                    )
                if use_exact_match:
                    exact_value_key = f'exact_retrieval_{split_name}'
                    exact_mask_key = f'exact_available_{split_name}'
                    if (
                        exact_value_key not in retrieval_cache
                        or exact_mask_key not in retrieval_cache
                    ):
                        raise KeyError(
                            'Exact-match cache is missing '
                            f'{exact_value_key!r} or {exact_mask_key!r}'
                        )
                    exact_values = np.asarray(
                        retrieval_cache[exact_value_key], dtype=np.float32
                    )
                    exact_available = np.asarray(
                        retrieval_cache[exact_mask_key], dtype=np.uint8
                    )
                    if not (
                        exact_values.shape == expected_shape
                        and exact_available.shape == expected_shape
                    ):
                        raise ValueError(
                            f'Exact-match {split_name} arrays must have '
                            f'shape {expected_shape}'
                        )
                    if not np.isfinite(exact_values).all():
                        raise ValueError(
                            f'{exact_value_key} contains non-finite values'
                        )
                    if not np.isin(exact_available, (0, 1)).all():
                        raise ValueError(
                            f'{exact_mask_key} must be binary'
                        )
                    exact_retrieval_by_split[split_name] = exact_values
                    exact_available_by_split[split_name] = exact_available
        logger.info(
            'Loaded train-only analog retrieval cache: %s '
            '(K=%s, train leave-one-out=True)',
            cache_path,
            getattr(args, 'analog_retrieval_k', 'unknown'),
        )
        if compressed_retrieval:
            logger.info(
                'Decoded train-only per-node temporal PCA retrieval codes '
                '(rank=%d); exact-history values remain uncompressed',
                compression_rank,
            )
        if quantized_retrieval:
            logger.info(
                'Decoded train-only symmetric int8 retrieval cache '
                '(per-horizon/node scales); exact-history values are stored '
                'separately'
            )
        if use_exact_match:
            logger.info(
                'Loaded exact-history continuation means '
                '(masked MSE=0, train leave-one-out=True)'
            )

    if use_candidate_reranker:
        candidate_cache_path = getattr(
            args, 'analog_candidate_cache', 'none'
        )
        if (
            candidate_cache_path in (None, '', 'none')
            or not os.path.exists(candidate_cache_path)
        ):
            raise FileNotFoundError(
                'Analog candidate index cache not found: '
                f'{candidate_cache_path}'
            )
        expected_indices = {
            'train': idx_train,
            'val': idx_val,
            'test': idx_test,
        }
        with np.load(
            candidate_cache_path, allow_pickle=False
        ) as candidate_cache:
            metadata_expectations = {
                'k': int(getattr(args, 'analog_retrieval_k', 4)),
                'input_length': int(args.input_length),
                'predict_length': int(args.predict_length),
                'min_valid_history': int(getattr(
                    args, 'analog_candidate_min_valid_history', 6
                )),
                'train_leave_one_out': 1,
                'candidates_train_only': 1,
            }
            for metadata_key, expected_value in metadata_expectations.items():
                if metadata_key not in candidate_cache:
                    raise KeyError(
                        'Candidate cache is missing metadata '
                        f'{metadata_key!r}'
                    )
                actual_value = int(np.asarray(
                    candidate_cache[metadata_key]
                ).item())
                if actual_value != expected_value:
                    raise ValueError(
                        f'Candidate cache {metadata_key}={actual_value}, '
                        f'expected {expected_value}'
                    )
            candidate_metric = str(getattr(
                args, 'analog_candidate_distance_metric', 'masked_mse'
            ))
            candidate_strategy = str(getattr(
                args, 'analog_candidate_selection_strategy', 'nearest'
            ))
            if candidate_metric not in (
                'masked_mse', 'masked_mse_trend', 'masked_mae',
                'masked_mae_trend_volatility'
            ):
                raise ValueError(
                    f'Unsupported candidate distance metric: {candidate_metric!r}'
                )
            if candidate_strategy not in ('nearest', 'rank_spread') and not (
                candidate_strategy.startswith('mmr')
                or candidate_strategy.startswith('learned_scorer')
            ):
                raise ValueError(
                    'Unsupported candidate selection strategy: '
                    f'{candidate_strategy!r}'
                )
            expected_rule = (
                'masked_history_' + candidate_metric.removeprefix('masked_')
            )
            expected_order = candidate_metric + (
                '_rank_spread'
                if candidate_strategy == 'rank_spread'
                else '_ascending'
            )
            text_metadata = {
                'candidate_cache_type': 'train_endpoint_indices',
                'calendar_mode': 'tod',
                'retrieval_rule': expected_rule,
                'candidate_order': expected_order,
            }
            for metadata_key, expected_value in text_metadata.items():
                if (
                    metadata_key not in candidate_cache
                    or str(np.asarray(
                        candidate_cache[metadata_key]
                    ).item()) != expected_value
                ):
                    raise ValueError(
                        f'Candidate cache {metadata_key} must be '
                        f'{expected_value!r}'
                    )
            if 'history_distance_metric' in candidate_cache:
                cached_metric = str(np.asarray(
                    candidate_cache['history_distance_metric']
                ).item())
                if cached_metric != candidate_metric:
                    raise ValueError(
                        'Candidate cache history_distance_metric='
                        f'{cached_metric!r}, expected {candidate_metric!r}'
                    )
            if 'candidate_selection_strategy' in candidate_cache:
                cached_strategy = str(np.asarray(
                    candidate_cache['candidate_selection_strategy']
                ).item())
                if cached_strategy != candidate_strategy:
                    raise ValueError(
                        'Candidate cache candidate_selection_strategy='
                        f'{cached_strategy!r}, expected {candidate_strategy!r}'
                    )
            cached_mean = np.asarray(candidate_cache['scaler_mean'])
            cached_std = np.asarray(candidate_cache['scaler_std'])
            if not (
                np.allclose(cached_mean, scaler_mean, rtol=0.0, atol=1e-7)
                and np.allclose(
                    cached_std, scaler_std, rtol=0.0, atol=1e-7
                )
            ):
                raise ValueError(
                    'Candidate cache scaler does not match active dataset'
                )
            train_membership = np.zeros(len(data), dtype=np.uint8)
            train_membership[idx_train] = 1
            for split_name, split_indices in expected_indices.items():
                idx_key = f'idx_{split_name}'
                value_key = f'candidate_index_{split_name}'
                if idx_key not in candidate_cache or value_key not in candidate_cache:
                    raise KeyError(
                        f'Candidate cache is missing {idx_key!r} or '
                        f'{value_key!r}'
                    )
                if not np.array_equal(
                    candidate_cache[idx_key], split_indices
                ):
                    raise ValueError(
                        f'Candidate {split_name} query indices do not match '
                        'the active split'
                    )
                candidate_index = np.asarray(
                    candidate_cache[value_key], dtype=np.int32
                )
                expected_shape = (
                    len(split_indices),
                    int(getattr(args, 'analog_retrieval_k', 4)),
                    num_nodes,
                )
                if candidate_index.shape != expected_shape:
                    raise ValueError(
                        f'Expected {value_key} shape {expected_shape}, got '
                        f'{candidate_index.shape}'
                    )
                valid = candidate_index >= 0
                if np.any(
                    candidate_index[valid] >= len(data)
                ) or np.any(
                    train_membership[candidate_index[valid]] != 1
                ):
                    raise ValueError(
                        f'{value_key} contains a non-training candidate'
                    )
                if split_name == 'train' and np.any(
                    candidate_index
                    == split_indices[:, None, None]
                ):
                    raise ValueError(
                        'Training candidate cache violates leave-one-out'
                    )
                safe_candidate = np.maximum(candidate_index, 0)
                same_tod = (
                    time_data[safe_candidate, 0]
                    == time_data[split_indices, 0][:, None, None]
                )
                if np.any(valid & ~same_tod):
                    raise ValueError(
                        f'{value_key} contains a different-TOD candidate'
                    )
                candidate_index_by_split[split_name] = candidate_index
        parent_prediction_cache_path = getattr(
            args, 'analog_candidate_parent_prediction_cache', 'none'
        )
        use_parent_residual_features = bool(getattr(
            args, 'analog_candidate_parent_residual_features', 0
        ))
        if use_parent_residual_features:
            if use_code_domain:
                raise ValueError(
                    'Candidate-parent residual features require full '
                    'candidate futures; disable code-domain aggregation'
                )
            if (
                parent_prediction_cache_path in (None, '', 'none')
                or not os.path.exists(parent_prediction_cache_path)
            ):
                raise FileNotFoundError(
                    'Candidate parent prediction cache not found: '
                    f'{parent_prediction_cache_path}'
                )
            with np.load(
                parent_prediction_cache_path, allow_pickle=False
            ) as parent_cache:
                if 'backbone_pred_train' not in parent_cache:
                    raise KeyError(
                        'Candidate parent cache must contain '
                        'backbone_pred_train'
                    )
                candidate_parent_prediction_train = np.asarray(
                    parent_cache['backbone_pred_train'], dtype=np.float32
                )
                expected_parent_shape = (
                    len(idx_train), args.predict_length,
                    num_nodes, args.pre_dim,
                )
                if (
                    candidate_parent_prediction_train.shape
                    != expected_parent_shape
                ):
                    raise ValueError(
                        'Candidate parent prediction shape mismatch: '
                        f'{candidate_parent_prediction_train.shape} vs '
                        f'{expected_parent_shape}'
                    )
                if not np.isfinite(
                    candidate_parent_prediction_train
                ).all():
                    raise ValueError(
                        'Candidate parent predictions contain non-finite values'
                    )
                for key, active in (
                    ('scaler_mean', scaler_mean),
                    ('scaler_std', scaler_std),
                ):
                    if key not in parent_cache or not np.allclose(
                        np.asarray(parent_cache[key]), np.asarray(active),
                        rtol=0.0, atol=1e-7,
                    ):
                        raise ValueError(
                            f'Candidate parent cache {key} does not match '
                            'the active dataset scaler'
                        )
            logger.info(
                'Loaded train-only candidate parent predictions for '
                'target-free residual features: %s shape=%s',
                parent_prediction_cache_path,
                candidate_parent_prediction_train.shape,
            )
        elif parent_prediction_cache_path not in (None, '', 'none'):
            raise ValueError(
                'analog_candidate_parent_prediction_cache was provided but '
                'analog_candidate_parent_residual_features is disabled'
            )
        if use_code_domain:
            code_cache_path = getattr(
                args, 'analog_candidate_code_cache', 'none'
            )
            if code_cache_path in (None, '', 'none') or not os.path.exists(code_cache_path):
                raise FileNotFoundError(
                    'Code-domain aggregation requires candidate code cache: '
                    f'{code_cache_path}'
                )
            with np.load(code_cache_path, allow_pickle=False) as code_cache:
                for key in ('candidate_code_rank', 'input_length', 'predict_length', 'k'):
                    if key not in code_cache:
                        raise KeyError(f'Candidate code cache is missing {key!r}')
                code_rank = int(np.asarray(code_cache['candidate_code_rank']).item())
                if int(np.asarray(code_cache['input_length']).item()) != int(args.input_length):
                    raise ValueError('Candidate code input_length does not match active config')
                if int(np.asarray(code_cache['predict_length']).item()) != int(args.predict_length):
                    raise ValueError('Candidate code predict_length does not match active config')
                if int(np.asarray(code_cache['k']).item()) != int(getattr(args, 'analog_retrieval_k', 4)):
                    raise ValueError('Candidate code K does not match active config')
                if 'retrieval_temporal_mean' not in code_cache or 'retrieval_temporal_basis' not in code_cache:
                    raise KeyError('Candidate code cache is missing PCA mean/basis')
                args.analog_candidate_code_rank = code_rank
                args.analog_candidate_code_mean = np.asarray(
                    code_cache['retrieval_temporal_mean'], dtype=np.float32
                )
                args.analog_candidate_code_basis = np.asarray(
                    code_cache['retrieval_temporal_basis'], dtype=np.float32
                )
                for split_name in ('train', 'val', 'test'):
                    key = f'candidate_code_{split_name}'
                    if key not in code_cache:
                        continue
                    idx_key = f'idx_{split_name}'
                    if idx_key not in code_cache or not np.array_equal(
                        code_cache[idx_key], expected_indices[split_name]
                    ):
                        raise ValueError(
                            f'Candidate code {split_name} indices do not match active split'
                        )
                    code = np.asarray(code_cache[key])
                    expected = (
                        len(expected_indices[split_name]),
                        int(getattr(args, 'analog_retrieval_k', 4)),
                        code_rank, num_nodes, args.pre_dim,
                    )
                    if code.shape != expected:
                        raise ValueError(
                            f'Expected {key} shape {expected}, got {code.shape}'
                        )
                    if not np.isfinite(code).all():
                        raise ValueError(f'{key} contains non-finite values')
                    candidate_code_by_split[split_name] = code
            if candidate_code_by_split['val'] is None:
                raise KeyError('Candidate code cache must include candidate_code_val')
            logger.info(
                'Loaded candidate PCA codes for code-domain aggregation: rank=%d, val_shape=%s',
                code_rank, candidate_code_by_split['val'].shape,
            )
        logger.info(
            'Loaded train-only K=%d candidate endpoint cache: %s '
            '(train leave-one-out=True, test targets unused)',
            int(getattr(args, 'analog_retrieval_k', 4)),
            candidate_cache_path,
        )

    if use_weekly_retrieval:
        weekly_cache_path = getattr(
            args, 'analog_weekly_retrieval_cache', 'none'
        )
        if (
            weekly_cache_path in (None, '', 'none')
            or not os.path.exists(weekly_cache_path)
        ):
            raise FileNotFoundError(
                'Weekly analog retrieval cache not found: '
                f'{weekly_cache_path}'
            )
        with np.load(
            weekly_cache_path, allow_pickle=False
        ) as weekly_cache:
            weekly_metadata = {
                'k': int(getattr(args, 'analog_weekly_retrieval_k', 3)),
                'input_length': int(args.input_length),
                'predict_length': int(args.predict_length),
                'train_leave_one_out': 1,
            }
            for metadata_key, expected_value in weekly_metadata.items():
                if metadata_key not in weekly_cache:
                    raise KeyError(
                        f'Weekly cache is missing {metadata_key!r}'
                    )
                actual_value = int(np.asarray(
                    weekly_cache[metadata_key]
                ).item())
                if actual_value != expected_value:
                    raise ValueError(
                        f'Weekly cache {metadata_key}={actual_value}, '
                        f'expected {expected_value}'
                    )
            if (
                'calendar_mode' not in weekly_cache
                or str(np.asarray(
                    weekly_cache['calendar_mode']
                ).item()) != 'tod_dow'
            ):
                raise ValueError(
                    'Weekly cache calendar_mode must be tod_dow'
                )
            weekly_mean = np.asarray(weekly_cache['scaler_mean'])
            weekly_std = np.asarray(weekly_cache['scaler_std'])
            if not (
                np.allclose(
                    weekly_mean, scaler_mean, rtol=0.0, atol=1e-7
                )
                and np.allclose(
                    weekly_std, scaler_std, rtol=0.0, atol=1e-7
                )
            ):
                raise ValueError(
                    'Weekly cache scaler does not match active dataset scaler'
                )
            expected_indices = {
                'train': idx_train,
                'val': idx_val,
                'test': idx_test,
            }
            for split_name, split_indices in expected_indices.items():
                idx_key = f'idx_{split_name}'
                value_key = f'retrieval_{split_name}'
                if (
                    idx_key not in weekly_cache
                    or value_key not in weekly_cache
                ):
                    raise KeyError(
                        f'Weekly cache is missing {idx_key!r} or '
                        f'{value_key!r}'
                    )
                if not np.array_equal(
                    weekly_cache[idx_key], split_indices
                ):
                    raise ValueError(
                        f'Weekly {split_name} indices do not match the '
                        'active dataset split'
                    )
                weekly_values = np.asarray(
                    weekly_cache[value_key], dtype=np.float32
                )
                expected_shape = (
                    len(split_indices),
                    args.predict_length,
                    num_nodes,
                    args.pre_dim,
                )
                if weekly_values.shape != expected_shape:
                    raise ValueError(
                        f'Expected weekly {value_key} shape '
                        f'{expected_shape}, got {weekly_values.shape}'
                    )
                if not np.isfinite(weekly_values).all():
                    raise ValueError(
                        f'Weekly {value_key} contains non-finite values'
                    )
                weekly_retrieval_by_split[split_name] = weekly_values
        logger.info(
            'Loaded train-only weekly analog cache: %s '
            '(same TOD+DOW, K=%s, train leave-one-out=True)',
            weekly_cache_path,
            getattr(args, 'analog_weekly_retrieval_k', 'unknown'),
        )

    if use_metric_retrieval:
        metric_cache_path = getattr(
            args, 'analog_metric_retrieval_cache', 'none'
        )
        if (
            metric_cache_path in (None, '', 'none')
            or not os.path.exists(metric_cache_path)
        ):
            raise FileNotFoundError(
                'Future-consistent metric cache not found: '
                f'{metric_cache_path}'
            )
        with np.load(
            metric_cache_path, allow_pickle=False
        ) as metric_cache:
            integer_metadata = {
                'k': int(getattr(args, 'analog_metric_retrieval_k', 4)),
                'input_length': int(args.input_length),
                'predict_length': int(args.predict_length),
                'train_leave_one_out': 1,
                'metric_train_only': 1,
            }
            for metadata_key, expected_value in integer_metadata.items():
                if metadata_key not in metric_cache:
                    raise KeyError(
                        f'Metric cache is missing {metadata_key!r}'
                    )
                actual_value = int(np.asarray(
                    metric_cache[metadata_key]
                ).item())
                if actual_value != expected_value:
                    raise ValueError(
                        f'Metric cache {metadata_key}={actual_value}, '
                        f'expected {expected_value}'
                    )
            if (
                'metric_type' not in metric_cache
                or str(np.asarray(
                    metric_cache['metric_type']
                ).item()) != 'future_supervised_mlp_v1'
            ):
                raise ValueError(
                    'Metric cache type must be future_supervised_mlp_v1'
                )
            metric_mean = np.asarray(metric_cache['scaler_mean'])
            metric_std = np.asarray(metric_cache['scaler_std'])
            if not (
                np.allclose(
                    metric_mean, scaler_mean, rtol=0.0, atol=1e-7
                )
                and np.allclose(
                    metric_std, scaler_std, rtol=0.0, atol=1e-7
                )
            ):
                raise ValueError(
                    'Metric cache scaler does not match active dataset scaler'
                )
            expected_indices = {
                'train': idx_train,
                'val': idx_val,
                'test': idx_test,
            }
            for split_name, split_indices in expected_indices.items():
                idx_key = f'idx_{split_name}'
                value_key = f'retrieval_{split_name}'
                if (
                    idx_key not in metric_cache
                    or value_key not in metric_cache
                ):
                    raise KeyError(
                        f'Metric cache is missing {idx_key!r} or '
                        f'{value_key!r}'
                    )
                if not np.array_equal(
                    metric_cache[idx_key], split_indices
                ):
                    raise ValueError(
                        f'Metric {split_name} indices do not match the '
                        'active dataset split'
                    )
                metric_values = np.asarray(
                    metric_cache[value_key], dtype=np.float32
                )
                expected_shape = (
                    len(split_indices),
                    args.predict_length,
                    num_nodes,
                    args.pre_dim,
                )
                if metric_values.shape != expected_shape:
                    raise ValueError(
                        f'Expected metric {value_key} shape '
                        f'{expected_shape}, got {metric_values.shape}'
                    )
                if not np.isfinite(metric_values).all():
                    raise ValueError(
                        f'Metric {value_key} contains non-finite values'
                    )
                metric_retrieval_by_split[split_name] = metric_values
        logger.info(
            'Loaded future-consistent train-memory cache: %s '
            '(K=%s, train leave-one-out=True, metric train-only=True)',
            metric_cache_path,
            getattr(args, 'analog_metric_retrieval_k', 'unknown'),
        )

    dataloader = {}
    # ===== KD soft targets (optional) =====
    teacher_predictions_by_split = {
        'train': None, 'val': None, 'test': None,
    }
    tp_path = getattr(args, 'teacher_predictions_path', 'none')
    kd_alpha = float(getattr(args, 'kd_alpha', 0.0))
    strict_teacher_alignment = bool(getattr(
        args, 'teacher_predictions_strict_alignment', 0
    ))
    required_teacher_splits = tuple(
        split.strip().lower()
        for split in str(getattr(
            args,
            'teacher_predictions_required_splits',
            'train,val,test',
        )).split(',')
        if split.strip()
    )
    invalid_teacher_splits = sorted(
        set(required_teacher_splits).difference({'train', 'val', 'test'})
    )
    if invalid_teacher_splits:
        raise ValueError(
            'teacher_predictions_required_splits contains invalid splits: '
            f'{invalid_teacher_splits}'
        )
    if kd_alpha > 0.0 and 'train' not in required_teacher_splits:
        raise ValueError(
            'kd_alpha > 0 requires train in '
            'teacher_predictions_required_splits'
        )
    if tp_path not in (None, '', 'none') and not os.path.exists(tp_path):
        raise FileNotFoundError(
            f'KD teacher prediction cache not found: {tp_path}'
        )
    if kd_alpha > 0.0 and tp_path in (None, '', 'none'):
        raise ValueError(
            'kd_alpha > 0 requires teacher_predictions_path'
        )
    if tp_path not in (None, '', 'none'):
        # Convenience: load the dump helper's npz, which splits predictions
        # per split. Legacy caches may be padded/truncated for backward
        # compatibility; paper-facing cross-teacher experiments enable strict
        # alignment and reject them instead of inventing soft targets.
        def _maybe_pad(arr, target_len):
            if arr is None or len(arr) == target_len:
                return arr
            if strict_teacher_alignment:
                raise ValueError(
                    'Strict teacher alignment requires exactly one prediction '
                    f'per split sample, got {len(arr)} != {target_len}'
                )
            if len(arr) == 0:
                raise ValueError('Teacher prediction split is empty')
            if len(arr) < target_len:
                pad_n = target_len - len(arr)
                padding = np.repeat(arr[-1:], pad_n, axis=0)
                return np.concatenate([arr, padding], axis=0)
            # Truncate when the dump helper used drop_last+pad_last_sample
            # for test, producing extra positional rows beyond idx_test.
            return arr[:target_len] 

        with np.load(tp_path, allow_pickle=False) as tp_cache:
            configured_space = str(getattr(
                args, 'teacher_predictions_space', 'normalized'
            )).lower()
            if 'prediction_space' in tp_cache:
                cached_space = str(np.asarray(
                    tp_cache['prediction_space']
                ).item()).lower()
                if cached_space != configured_space:
                    raise ValueError(
                        'Teacher cache prediction_space does not match the '
                        f'configuration: {cached_space!r} != '
                        f'{configured_space!r}'
                    )
            for split in ('train', 'val', 'test'):
                key = f'teacher_pred_{split}'
                if key not in tp_cache:
                    if (strict_teacher_alignment and split in required_teacher_splits) or (
                        kd_alpha > 0.0 and split == 'train'
                    ):
                        raise KeyError(
                            f'KD teacher cache is missing required key {key!r}'
                        )
                    continue
                arr = np.asarray(tp_cache[key], dtype=np.float32)
                expected_shape_tail = (
                    args.predict_length, num_nodes, args.pre_dim
                )
                if arr.shape[1:] != expected_shape_tail:
                    raise ValueError(
                        f'teacher_pred_{split} shape {arr.shape} incompatible '
                        f'with expected tail {expected_shape_tail}'
                    )
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f'teacher_pred_{split} contains non-finite values'
                    )
                if split == 'train':
                    arr = _maybe_pad(arr, idx_train.shape[0])
                elif split == 'val':
                    arr = _maybe_pad(arr, idx_val.shape[0])
                else:
                    arr = _maybe_pad(arr, idx_test.shape[0])
                teacher_predictions_by_split[split] = arr
                logger.info(
                    f'Loaded KD teacher predictions for {split}: '
                    f'shape={arr.shape}, teacher='
                    f'{getattr(args, "teacher_name", "unspecified")}, '
                    f'space={configured_space}, strict='
                    f'{strict_teacher_alignment}'
                )

    logger.info('creating train_loader ...')
    dataloader['train_loader'] = DataLoader_time(data, time_data, idx_train,
                                                 args.input_length, args.predict_length,
                                                 args.batch_size, logger, scaler = scaler,
                                                 retrieval_data=retrieval_by_split['train'],
                                                 weekly_retrieval_data=weekly_retrieval_by_split['train'],
                                                 metric_retrieval_data=metric_retrieval_by_split['train'],
                                                 retrieval_quality_data=retrieval_quality_by_split['train'],
                                                  exact_retrieval_data=exact_retrieval_by_split['train'],
                                                  exact_available_data=exact_available_by_split['train'],
                                                  analog_candidate_index_data=candidate_index_by_split['train'],
                                                  analog_candidate_code_data=candidate_code_by_split['train'],
                                                  analog_candidate_parent_prediction_data=candidate_parent_prediction_train,
                                                  analog_candidate_parent_index_data=(idx_train if candidate_parent_prediction_train is not None else None),
                                                  teacher_predictions_data=teacher_predictions_by_split['train'])
    logger.info('creating val_loader ...')
    dataloader['val_loader'] = DataLoader_time(
        data, time_data, idx_val,
        args.input_length, args.predict_length,
        args.batch_size, logger, scaler=scaler, drop_last=False,
        retrieval_data=retrieval_by_split['val'],
        weekly_retrieval_data=weekly_retrieval_by_split['val'],
        metric_retrieval_data=metric_retrieval_by_split['val'],
        retrieval_quality_data=retrieval_quality_by_split['val'],
        exact_retrieval_data=exact_retrieval_by_split['val'],
        exact_available_data=exact_available_by_split['val'],
        analog_candidate_index_data=candidate_index_by_split['val'],
        analog_candidate_code_data=candidate_code_by_split['val'],
        analog_candidate_parent_prediction_data=candidate_parent_prediction_train,
        analog_candidate_parent_index_data=(idx_train if candidate_parent_prediction_train is not None else None),
        teacher_predictions_data=teacher_predictions_by_split['val'],
    )
    logger.info('creating test_loader ...')
    num_test_samples = idx_test.shape[0]
    dataloader['test_loader'] = DataLoader_time(data,time_data, idx_test,
                                                args.input_length, args.predict_length,
                                                args.batch_size, logger, pad_last_sample=True, scaler= scaler,
                                                retrieval_data=retrieval_by_split['test'],
                                                weekly_retrieval_data=weekly_retrieval_by_split['test'],
                                                metric_retrieval_data=metric_retrieval_by_split['test'],
                                                retrieval_quality_data=retrieval_quality_by_split['test'],
                                                 exact_retrieval_data=exact_retrieval_by_split['test'],
                                                 exact_available_data=exact_available_by_split['test'],
                                                 analog_candidate_index_data=candidate_index_by_split['test'],
                                                 analog_candidate_code_data=candidate_code_by_split['test'],
                                                 analog_candidate_parent_prediction_data=candidate_parent_prediction_train,
                                                 analog_candidate_parent_index_data=(idx_train if candidate_parent_prediction_train is not None else None),
                                                 teacher_predictions_data=teacher_predictions_by_split['test'])

    return dataloader, num_nodes, scaler, num_test_samples


def load_dataset_for_analysis(dataset_name, input_length, predict_length, feat_dim):
    raw_data_path = file_path_dict[dataset_name]
    time_path = f"datasets/{dataset_name}/{dataset_name}_time.npz"
    index_path = f"datasets/{dataset_name}/{dataset_name}_{input_length}to{predict_length}_index_scaler.npz"

    print(f'loading raw data of {dataset_name}...')
    data, df_time = load_raw_data(raw_data_path)
    data = data[...,feat_dim]
    print(f'Data shape:{data.shape}')
    print(os.path.exists(time_path), time_path)
    print(os.path.exists(index_path), index_path)
    
    if not (os.path.exists(time_path) and os.path.exists(index_path)):
        raise NotImplementedError('no cache files!')
        # logger.info('creating cache files....')
        # time_data, idx_train, idx_val, idx_test, scaler_mean, scaler_std = create_dataset_files(args, data, df_time)
    else:
        print('loading cache files....')
        time_data = np.load(time_path)['time']
        index_scaler = np.load(index_path)
        idx_train = index_scaler['idx_train']
        idx_val = index_scaler['idx_val']
        idx_test = index_scaler['idx_test']
        scaler_mean = index_scaler['scaler_mean']
        scaler_std = index_scaler['scaler_std']

    print(f'Time Data shape: {time_data.shape}')
    num_nodes = data.shape[1]

    scaler = StandardScaler(mean=scaler_mean, std=scaler_std)

    return data, idx_train, idx_val, idx_test, scaler
