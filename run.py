import torch
import numpy as np
import argparse
import time

import csv
from torch import nn
import torch.optim as optim
from models import TLAST
from models import Scope

import os
import sys
from functools import partial

import logging
from logging import getLogger
import pickle
import json

from utils import utils, datasets_utils

parser = argparse.ArgumentParser()

# load configuration from files
parser.add_argument('--config_file', type=str, default='none')
parser.add_argument('--load_stdmae_ckpt', type=str, default='none',
                    help='Load STD-MAE pretrained encoder weights before training')

# the information about the data preprocessing
parser.add_argument('--dataset_name', type=str, default='PEMS08')
parser.add_argument('--input_length', type=int, default=-1)
parser.add_argument('--predict_length', type=int, default=-1)
parser.add_argument('--scaler_type', type=str,default='zscore')
parser.add_argument('--slice_size_per_day', type=int, default=-1)
parser.add_argument(
    '--index_scaler_path',
    type=str,
    default='none',
    help=(
        'Optional explicit index/scaler cache. This is used by train-only '
        'cross-fitting splits without overwriting the official dataset cache.'
    ),
)

# the hyper-parameter-setting in the model
parser.add_argument('--modelid', type=str, default='Scope')
parser.add_argument('--hid_dim', type=int, default=-1) # d in the paper
parser.add_argument('--d_out', type=int, default=-1) # d'
parser.add_argument(
    '--predictor_factor_rank', type=int, default=256,
    help=(
        'Rank of predictor_type=richer_factorized. Ignored by other '
        'predictors.'
    ),
)
parser.add_argument(
    '--predictor_activation_covariance', type=str, default='none',
    help=(
        'Optional .npz containing train-only predictor activation '
        'second_moment for activation-aware low-rank initialization.'
    ),
)
parser.add_argument(
    '--predictor_activation_ridge', type=float, default=1e-5,
    help='Relative diagonal ridge used by activation-aware factorization.',
)
parser.add_argument('--use_elastic_rank_router', type=int, default=0)
parser.add_argument('--elastic_low_rank', type=int, default=64)
parser.add_argument('--elastic_router_hidden', type=int, default=8)
parser.add_argument('--elastic_route_mode', type=str, default='batch_topk')
parser.add_argument('--elastic_rank_temperature', type=float, default=1.0)
parser.add_argument('--elastic_initial_high_probability', type=float, default=0.8)
parser.add_argument('--elastic_target_high_fraction', type=float, default=0.5)
parser.add_argument('--elastic_budget_loss_weight', type=float, default=1.0)
parser.add_argument('--elastic_low_consistency_weight', type=float, default=0.2)
parser.add_argument(
    '--use_signed_increment_mixer', type=int, default=0,
    help=(
        'For richer_factorized, couple signed step increments across local '
        'forecast horizons with an identity-initialized structured decoder.'
    ),
)
parser.add_argument(
    '--signed_increment_mixer_radius', type=int, default=2,
    help='Local horizon radius used by the signed-increment mixer.',
)
parser.add_argument(
    '--signed_increment_mixer_max_scale', type=float, default=0.25,
    help='Maximum absolute learned coefficient added to the identity mixer.',
)
parser.add_argument(
    '--use_signed_change_encoding', type=int, default=0,
    help=(
        'Add a zero-gated first-difference input encoder alongside the '
        'ordinary traffic-level encoder (0/1).'
    ),
)
parser.add_argument(
    '--signed_change_encoding_max_scale', type=float, default=0.5,
    help='Maximum per-channel fusion scale for signed change features.',
)
parser.add_argument('--use_short_window_dynamics_encoding', type=int, default=0)
parser.add_argument('--short_window_dynamics_encoding_max_scale', type=float, default=0.5)
parser.add_argument('--n_heads', type=int, default=-1) # h, the number of heads
parser.add_argument('--M', type=int, default=-1) # m, the number of proxy nodes
parser.add_argument('--num_layers', type=int, default=-1) # L
parser.add_argument('--tau', type=int, default=3, help='0:no TCN') # \tau
parser.add_argument('--enc_dropout', type=float, default=0.1) # \phi, dropout rate
parser.add_argument('--att_dropout', type=float, default=0.1)
parser.add_argument('--emb_dropout', type=float, default=0.1)
parser.add_argument('--te_emb_dropout', type=float, default=0.1)
parser.add_argument('--se_emb_dropout', type=float, default=0.1)
parser.add_argument('--tle_emb_dropout', type=float, default=0.1)

# Graph Convolution parameters
parser.add_argument('--use_graph_conv', type=int, default=0, help='whether to use graph convolution (0/1)')
parser.add_argument('--graph_conv_type', type=str, default='adaptive', help='graph convolution type: adaptive/multiscale')
parser.add_argument('--k_hops', type=str, default='1,2', help='k-hops for multiscale graph conv, e.g., "1,2"')
parser.add_argument('--pearson_adj_path', type=str, default=None, help='path to Pearson correlation adjacency matrix')

# MAPA（记忆增强原型注意力）相关参数
parser.add_argument('--lambda_ortho', type=float, default=0.01, 
                    help='正交损失权重（orthogonal loss weight for MAPA）')
parser.add_argument('--num_prototypes', type=int, default=16,
                    help='Memory Bank 中原型数量（number of prototypes in Memory Bank）')

# FTLE（全时滞嵌入）和 MS-TCN 相关参数
parser.add_argument('--use_ftle', type=int, default=1,
                    help='是否使用 FTLE（0/1）')
parser.add_argument(
    '--use_time_lag_embedding', type=int, default=1,
    help=(
        'Whether to add any time-lag embedding at all (0/1). Note that '
        'use_ftle=0 still means the original last-reference TLE.'
    ),
)
parser.add_argument(
    '--single_tle_mode', type=str, default='last',
    choices=['last', 'selective'],
    help=(
        'use_ftle=0 时的单参考 TLE 模式：last=仅最后时刻参考；'
        'selective=从最后时刻参考精确初始化，并用零初始化门选择更早参考'
    ),
)
parser.add_argument('--use_mapa', type=int, default=1,
                    help='是否使用 MAPA（0/1）')
parser.add_argument('--tau_list', type=str, default=None,
                    help='MS-TCN 的 tau 列表（comma-separated，例如 "3,5,7"）')

parser.add_argument('--return_att', type=int, default=0)
parser.add_argument('--att_type', type=str, default='proxy')
parser.add_argument('--norm_flag', type=str, default='none')
parser.add_argument('--use_revin', type=int, default=0)
parser.add_argument('--activation_data', type=str, default='relu')
parser.add_argument('--activation_enc', type=str, default='gelu')
parser.add_argument('--activation_dec', type=str, default='gelu')
parser.add_argument('--query_cross_time', type=int, default=1)

# the hyper-parameter-setting of the training process
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float,default=0.001)
parser.add_argument(
    '--neighbor_learning_rate', type=float, default=None,
    help='Optional learning rate for sparse_neighbor_innovation parameters.',
)
parser.add_argument(
    '--reranker_learning_rate', type=float, default=None,
    help='Optional learning rate for the analog candidate reranker',
)
parser.add_argument(
    '--lowrank_learning_rate', type=float, default=None,
    help='Optional learning rate for the reopened analog lowrank adapter',
)
parser.add_argument(
    '--analog_utility_loss_weight', type=float, default=0.0,
    help=(
        'Weight of train-only listwise candidate-utility supervision; '
        'validation/Test never use targets in the reranker'
    ),
)
parser.add_argument(
    '--analog_utility_temperature', type=float, default=1.0,
    help='Temperature after per-position standardization of candidate utility',
)
parser.add_argument('--weight_decay', type=float,default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument(
    '--early_stop',
    type=lambda value: str(value).lower() in ('1', 'true', 'yes', 'y'),
    default=False,
)
parser.add_argument('--early_stop_step', type=int, default=20)
parser.add_argument('--early_stop_start_epoch', type=int, default=50)
parser.add_argument(
    '--validation_only',
    type=lambda value: str(value).lower() in ('1', 'true', 'yes', 'y'),
    default=False,
    help='Finish after model selection on validation; do not evaluate Test.',
)
parser.add_argument('--lr_decay', type=int, default=1)
parser.add_argument('--huber_delta', type=int, default=2, help='delta in huber loss')
parser.add_argument('--loss_type', type=str, default="huber")
parser.add_argument('--time_weights', type=str, default=None,
                    help='Time step weights for weighted loss. Format: "1.0,1.1,1.2,...,2.0" (comma-separated)')
parser.add_argument('--horizon_grad_balance_beta', type=float, default=0.0,
                    help='Dynamic ordered horizon gradient-balancing strength; 0 disables it.')
parser.add_argument('--horizon_grad_balance_momentum', type=float, default=0.9)
parser.add_argument('--horizon_grad_balance_interval', type=int, default=50)
parser.add_argument('--horizon_grad_balance_warmup_steps', type=int, default=100)
parser.add_argument('--horizon_grad_balance_min_weight', type=float, default=0.5)
parser.add_argument('--horizon_grad_balance_max_weight', type=float, default=1.5)
parser.add_argument(
    '--diff_consistency_weight', type=float, default=0.0,
    help=(
        '相邻预测步差分一致性损失权重；0=关闭。'
        '损失仅使用真实值相邻端点都非零的位置'
    ),
)
parser.add_argument(
    '--volatility_mae_strength', type=float, default=0.0,
    help=(
        '训练期历史高波动样本/节点的额外MAE权重；0=关闭，'
        '验证和Test始终使用普通masked MAE'
    ),
)
parser.add_argument(
    '--volatility_mae_quantile', type=float, default=0.75,
    help='每个训练batch内判定高波动样本/节点的历史波动分位数',
)
parser.add_argument(
    '--node_tail_risk_strength', type=float, default=0.0,
    help=(
        'Blend weight for a Train-only node-level CVaR MAE objective. '
        '0 disables it; validation/Test always use ordinary masked MAE.'
    ),
)
parser.add_argument(
    '--node_tail_risk_fraction', type=float, default=0.1,
    help=(
        'Fraction of nodes with the largest masked MAE in each Train batch '
        'used by the node-level CVaR objective.'
    ),
)
parser.add_argument(
    '--node_tail_risk_scale_path', type=str, default='none',
    help=(
        'Optional Train-only .npz containing node_scales. When provided, '
        'CVaR ranks nodes by MAE divided by their Train-fitted scale.'
    ),
)
parser.add_argument(
    '--future_change_balance_strength', type=float, default=0.0,
    help=(
        'Extra training MAE weight for sample/node pairs whose supervised '
        'future change score exceeds a fixed Train-only per-node threshold. '
        'Validation/Test always use ordinary masked MAE.'
    ),
)
parser.add_argument(
    '--future_change_balance_stats_path', type=str, default='none',
    help='Train-only .npz containing node_thresholds for change balancing.',
)
parser.add_argument(
    '--change_persistence_loss_weight', type=float, default=0.0,
    help=(
        'Training-only log-change-magnitude loss on positions whose observed '
        'history exceeds a fixed Train-only per-node threshold; 0=disabled.'
    ),
)
parser.add_argument(
    '--change_persistence_stats_path', type=str, default='none',
    help=(
        'Train-only .npz containing history_node_thresholds for change '
        'persistence supervision.'
    ),
)
parser.add_argument(
    '--change_event_loss_weight', type=float, default=0.0,
    help=(
        'Training-only signed per-horizon transition loss on positions with '
        'active observed histories; 0=disabled.'
    ),
)
parser.add_argument(
    '--change_event_emphasis', type=float, default=2.0,
    help='Extra relative weight assigned to large true transition events.',
)
parser.add_argument(
    '--change_event_stats_path', type=str, default='none',
    help=(
        'Train-only .npz containing history_node_thresholds for signed-event '
        'supervision.'
    ),
)

parser.add_argument('--eval_mask', type=int, default=0,
                    help='eval_mask; {-1: not mask; 0: exclude zero; x: exclude values below x}')
parser.add_argument('--lr_scheduler_type', type=str, default='cosinelr')
parser.add_argument('--save_output', type=int, default=0)
parser.add_argument(
    '--save_validation_output', type=int, default=0,
    help=(
        'For validation-only/load_expid runs, save raw-scale validation '
        'predictions and truths without evaluating Test (0/1).'
    ),
)
parser.add_argument(
    '--use_student_risk_head', type=int, default=0,
    help='Attach a tiny single-Student sample-risk head (0/1).',
)
parser.add_argument(
    '--student_risk_hidden', type=int, default=16,
    help='Hidden width of the single-Student risk head.',
)
parser.add_argument(
    '--student_risk_loss_weight', type=float, default=0.0,
    help='Weight for Train-only log absolute-error regression.',
)
parser.add_argument('--seed', type=int, default=0,
                    help='随机种子。0=不固定（保持原默认行为）；>0=固定 torch/numpy/random 并设置 cudnn 确定性，用于可复现训练与 Seed Ensemble')
parser.add_argument('--use_ema', type=int, default=0,
                    help='EMA 指数滑动平均权重（0/1）。开启后验证/保存均使用 EMA 权重。')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='EMA 衰减系数（默认 0.999）')
parser.add_argument('--use_swa', type=int, default=0,
                    help='SWA 随机权重平均（0/1）。开启后从 --swa_start_epoch 起平均权重。')
parser.add_argument('--swa_start_epoch', type=int, default=50,
                    help='SWA 开始平均的 epoch（默认 50）')
parser.add_argument('--use_node_dropout', type=int, default=0,
                    help='节点/传感器级 dropout（0/1）。开启后训练时按概率整条屏蔽部分传感器，'
                         '迫使模型从空间邻域重建，降低对单一传感器的过拟合。仅训练时生效，eval 自动关闭。')
parser.add_argument('--node_dropout_p', type=float, default=0.1,
                    help='Node Dropout 屏蔽比例')
parser.add_argument('--use_parallel_node_emb', type=int, default=0,
                    help='2B: 在 MAPA 之后并行加可学习节点嵌入，补回原型压缩丢失的节点身份 (0/1)')
parser.add_argument('--parallel_node_emb_dropout', type=float, default=0.1,
                    help='2B 并行节点嵌入的 dropout 概率')
parser.add_argument('--use_future_time_decoder', type=int, default=0,
                    help='用已知未来时刻/星期特征调制 richer predictor (0/1)')
parser.add_argument(
    '--use_volatility_delta_scale', type=int, default=0,
    help='按可观察历史波动动态缩放相对最后观测值的预测增量 (0/1)',
)
parser.add_argument('--volatility_delta_hidden', type=int, default=8)
parser.add_argument('--volatility_delta_max', type=float, default=0.1)
parser.add_argument(
    '--use_node_linear_dynamics', type=int, default=0,
    help=(
        'Add a zero-initialized per-node/per-horizon linear bypass from the '
        'observable 12-step relative trajectory to the forecast delta (0/1)'
    ),
)
parser.add_argument(
    '--node_linear_dynamics_scale', type=float, default=0.25,
    help='Fixed shrinkage applied to the node-linear dynamics correction',
)
parser.add_argument(
    '--use_regime_gated_residual', type=int, default=0,
    help=(
        'Add a zero-initialized shared nonlinear residual adapter conditioned '
        'on observed change regimes and the parent forecast delta (0/1)'
    ),
)
parser.add_argument(
    '--multires_block_loss_weight', type=float, default=0.0,
    help=(
        'Training-only masked MAE on consecutive future block means; '
        'validation/Test remain standard per-horizon MAE'
    ),
)
parser.add_argument('--regime_residual_hidden', type=int, default=32)
parser.add_argument('--regime_residual_horizon_dim', type=int, default=8)
parser.add_argument('--regime_residual_scale', type=float, default=0.25)
parser.add_argument(
    '--use_multires_block_residual', type=int, default=0,
    help=(
        'Add a zero-initialized coarse future-block residual adapter (0/1)'
    ),
)
parser.add_argument('--multires_block_size', type=int, default=3)
parser.add_argument('--multires_block_hidden', type=int, default=32)
parser.add_argument('--multires_block_scale', type=float, default=0.25)
parser.add_argument(
    '--use_seasonal_drift_residual', type=int, default=0,
    help=(
        'Add a zero-initialized training-split TOD seasonal-drift residual '
        'to the parent forecast (0/1)'
    ),
)
parser.add_argument('--seasonal_drift_max', type=float, default=1.0)
parser.add_argument(
    '--use_analog_retrieval_residual', type=int, default=0,
    help=(
        'Blend the parent toward a train-memory KNN analog forecast using '
        'zero-initialized per-horizon coefficients (0/1)'
    ),
)
parser.add_argument('--analog_retrieval_max', type=float, default=1.0)
parser.add_argument('--analog_retrieval_k', type=int, default=4)
parser.add_argument(
    '--use_analog_lowrank_retrieval_residual', type=int, default=0,
    help=(
        'Use a zero-initialized rank-constrained node/horizon analog blend '
        '(0/1)'
    ),
)
parser.add_argument('--analog_lowrank_rank', type=int, default=2)
parser.add_argument(
    '--use_analog_temporal_lowrank_decoder', type=int, default=0,
    help=(
        'Apply a zero-initialized shared low-rank temporal correction to the '
        'decoded near-match retrieval sequence (0/1)'
    ),
)
parser.add_argument('--analog_temporal_decoder_rank', type=int, default=4)
parser.add_argument(
    '--analog_temporal_decoder_max_scale', type=float, default=0.25
)
parser.add_argument(
    '--use_analog_temporal_gap_residual', type=int, default=0,
    help=(
        'Apply a zero-initialized node-aware low-rank temporal correction '
        'after exact/confidence retrieval refinement (0/1)'
    ),
)
parser.add_argument('--analog_temporal_gap_rank', type=int, default=4)
parser.add_argument('--analog_temporal_gap_max', type=float, default=0.25)
parser.add_argument(
    '--use_analog_node_horizon_retrieval_residual', type=int, default=0,
    help=(
        'Use an independent zero-initialized analog blend coefficient for '
        'every forecast-step/sensor pair (0/1)'
    ),
)
parser.add_argument(
    '--analog_retrieval_aggregation', type=str, default='mean',
    choices=('mean', 'median', 'inverse_distance'),
)
parser.add_argument(
    '--use_analog_exact_match_residual', type=int, default=0,
    help=(
        'Add a zero-initialized horizon blend toward exact-history '
        'train-memory continuation means (0/1)'
    ),
)
parser.add_argument('--analog_exact_match_max', type=float, default=1.0)
parser.add_argument(
    '--analog_exact_min_valid_history', type=int, default=6,
)
parser.add_argument(
    '--analog_retrieval_cache', type=str, default='none',
    help='Precomputed train-only analog forecasts for train/val/test queries',
)
parser.add_argument(
    '--use_analog_candidate_reranker', type=int, default=0,
    help=(
        'Reweight the fixed K train-only candidates separately for every '
        'horizon while preserving the original mean at initialization (0/1)'
    ),
)
parser.add_argument(
    '--use_analog_direct_prediction', type=int, default=0,
    help=(
        'Use the reranked train-memory continuation directly as the final '
        'prediction, bypassing residual refinement after reranking (0/1)'
    ),
)
parser.add_argument('--analog_candidate_feature_dim', type=int, default=8)
parser.add_argument('--analog_candidate_disagreement_dim', type=int, default=6,
                    help='Number of per-candidate disagreement feature channels')
parser.add_argument('--analog_candidate_hidden', type=int, default=16)
parser.add_argument(
    '--analog_candidate_parent_residual_features', type=int, default=0,
    help=(
        'Add target-free per-candidate parent forecast/residual features to '
        'the reranker (0/1)'
    ),
)
parser.add_argument(
    '--analog_candidate_parent_prediction_cache', type=str, default='none',
    help=(
        'Frozen parent predictions containing backbone_pred_train, used '
        'only when candidate-parent residual features are enabled'
    ),
)
parser.add_argument(
    '--analog_candidate_level_align', type=int, default=0,
    help=(
        'Shift each candidate future by the observable current-vs-history '
        'last-value difference before reranking (0/1)'
    ),
)
parser.add_argument(
    '--analog_candidate_distance_metric', type=str, default='masked_mse',
    choices=('masked_mse', 'masked_mae', 'masked_mae_trend_volatility'),
    help=(
        'Observable history distance used when building the fixed-K '
        'candidate endpoint cache.'
    ),
)
parser.add_argument(
    '--analog_candidate_selection_strategy', type=str, default='nearest',
    choices=('nearest', 'rank_spread'),
    help='Selection strategy used by the fixed-K candidate cache builder.',
)
parser.add_argument(
    '--use_analog_code_domain_aggregation', type=int, default=0,
    help='Aggregate fixed candidate PCA codes and decode the final mixture once (0/1)',
)
parser.add_argument(
    '--analog_candidate_code_cache', type=str, default='none',
    help='Candidate PCA-code cache used by code-domain aggregation',
)
parser.add_argument(
    '--analog_code_domain_decode_candidates', type=int, default=1,
    help='Decode K candidate codes for disagreement scoring; 0 uses history features only',
)
parser.add_argument(
    '--use_analog_correction_utility_gate', type=int, default=0,
    help=(
        'Learn a horizon/node utility gate over the frozen near-match '
        'low-rank correction (0/1)'
    ),
)
parser.add_argument('--analog_correction_utility_hidden', type=int, default=8)
parser.add_argument('--analog_correction_utility_node_rank', type=int, default=2)
parser.add_argument('--analog_correction_utility_max', type=float, default=1.0)
parser.add_argument(
    '--use_analog_adaptive_residual_mixer', type=int, default=0,
    help=(
        'Use an ARM-style query/candidate residual mixer with a parent skip '
        'connection (0/1)'
    ),
)
parser.add_argument('--analog_adaptive_mixer_hidden', type=int, default=16)
parser.add_argument('--analog_adaptive_mixer_node_rank', type=int, default=2)
parser.add_argument('--analog_adaptive_mixer_max', type=float, default=0.25)
parser.add_argument(
    '--analog_candidate_min_valid_history', type=int, default=6
)
parser.add_argument(
    '--analog_candidate_cache', type=str, default='none',
    help='Train-only endpoint indices for the fixed K analog candidates',
)
parser.add_argument(
    '--use_analog_query_conditioned_residual', type=int, default=0,
    help='Condition candidate residual content on observable query-history differences (0/1)',
)
parser.add_argument('--analog_query_conditioned_hidden', type=int, default=8)
parser.add_argument('--analog_query_conditioned_max', type=float, default=0.25)
parser.add_argument(
    '--freeze_parent_train_analog_query_conditioned_residual', type=int, default=0,
    help='Train only the query-conditioned candidate residual on a frozen parent (0/1)',
)
parser.add_argument(
    '--freeze_parent_train_node_linear', type=int, default=0,
    help=(
        'Load a trained parent checkpoint, keep the parent in eval mode, and '
        'optimize only node_linear_dynamics (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_regime_residual', type=int, default=0,
    help=(
        'Load a trained parent checkpoint, keep the parent in eval mode, and '
        'optimize only regime_gated_residual (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_multires_block_residual', type=int, default=0,
    help=(
        'Load a trained parent checkpoint, keep the parent in eval mode, and '
        'optimize only multires_block_residual (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_seasonal_drift', type=int, default=0,
    help=(
        'Load a trained parent checkpoint, keep the parent in eval mode, and '
        'optimize only seasonal_drift_residual (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_retrieval', type=int, default=0,
    help=(
        'Load a trained parent checkpoint and optimize only '
        'analog_retrieval_residual (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_lowrank_retrieval', type=int, default=0,
    help=(
        'Load a trained parent checkpoint and optimize only '
        'analog_lowrank_retrieval_residual (0/1)'
    ),
)
parser.add_argument(
    '--use_analog_confidence_aware_retrieval_residual', type=int, default=0,
    help=(
        'Enable a zero-initialized local confidence refinement after the '
        'rank-2 analog (and optional exact-history) parent. Requires quality '
        'channels in the analog retrieval cache (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_candidate_reranker', type=int, default=0,
    help=(
        'Load the validated final checkpoint and optimize only the '
        'horizon-specific candidate reranker (0/1)'
    ),
)
parser.add_argument(
    '--joint_train_lowrank_with_reranker', type=int, default=0,
    help=(
        'When --freeze_parent_train_analog_candidate_reranker=1, also '
        'unfreeze the rank-2 lowrank adapter (analog_lowrank_retrieval_'
        'residual) so the two adapters are jointly optimized on top of '
        'the loaded parent checkpoint (0/1). Both adapters are '
        'initialized from the parent checkpoint so training starts from '
        'the validated result.'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_node_horizon_retrieval',
    type=int,
    default=0,
    help=(
        'Load a trained parent checkpoint and optimize only '
        'analog_node_horizon_retrieval_residual (0/1)'
    ),
)
parser.add_argument('--analog_confidence_max', type=float, default=0.25)
parser.add_argument(
    '--analog_confidence_node_rank', type=int, default=0,
    help=(
        'Low-rank node-aware confidence gate rank; 0 keeps the shared '
        'horizon-only gate (default).'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_correction_utility_gate', type=int, default=0,
    help=(
        'Load the validated parent and optimize only the correction utility '
        'gate (0/1)'
    ),
)
parser.add_argument(
    '--analog_confidence_feature_dim', type=int, default=3,
    choices=(3, 6),
    help='Confidence features: 3 legacy quality signals or 6 with forecast gaps.',
)
parser.add_argument(
    '--use_analog_weekly_retrieval_residual', type=int, default=0,
    help=(
        'Add a zero-initialized low-rank blend toward train-memory '
        'continuations matched by both TOD and DOW (0/1)'
    ),
)
parser.add_argument('--analog_weekly_retrieval_max', type=float, default=0.5)
parser.add_argument('--analog_weekly_rank', type=int, default=2)
parser.add_argument('--analog_weekly_retrieval_k', type=int, default=3)
parser.add_argument(
    '--analog_weekly_retrieval_cache', type=str, default='none',
)
parser.add_argument(
    '--use_analog_metric_retrieval_residual', type=int, default=0,
    help=(
        'Add a zero-initialized low-rank blend toward continuations selected '
        'by a train-only future-consistent history metric (0/1)'
    ),
)
parser.add_argument('--analog_metric_retrieval_max', type=float, default=0.5)
parser.add_argument('--analog_metric_rank', type=int, default=2)
parser.add_argument('--analog_metric_retrieval_k', type=int, default=4)
parser.add_argument(
    '--analog_metric_retrieval_cache', type=str, default='none',
)
parser.add_argument(
    '--freeze_parent_train_analog_confidence_aware_retrieval', type=int, default=0,
    help=(
        'Load a trained parent checkpoint and optimize only '
        'analog_confidence_aware_retrieval_residual (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_weekly_retrieval', type=int, default=0,
    help=(
        'Load a trained daily+exact parent checkpoint and optimize only '
        'analog_weekly_retrieval_residual (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_metric_retrieval', type=int, default=0,
    help=(
        'Load the daily+exact parent and optimize only '
        'analog_metric_retrieval_residual (0/1)'
    ),
)
parser.add_argument(
    '--teacher_predictions_path', type=str, default='none',
    help=(
        'Optional .npz with keys teacher_pred_{train,val,test} containing '
        'precomputed teacher predictions for knowledge distillation. Use '
        '"none" to disable.'
    ),
)
parser.add_argument(
    '--teacher_predictions_space', type=str, default='normalized',
    choices=('normalized', 'raw'),
    help=(
        'Value space used by teacher_predictions_path. "normalized" means '
        'the dataset z-score space; "raw" means the original traffic-flow '
        'space.'
    ),
)
parser.add_argument(
    '--teacher_name', type=str, default='unspecified',
    help='Human-readable teacher identity recorded in logs and configs.',
)
parser.add_argument(
    '--teacher_predictions_strict_alignment', type=int, default=0,
    help=(
        'Require every teacher split to have exactly one prediction per '
        'official split index and reject legacy padded/truncated caches (0/1).'
    ),
)
parser.add_argument(
    '--teacher_predictions_required_splits', type=str,
    default='train,val,test',
    help=(
        'Comma-separated teacher-cache splits required under strict '
        'alignment. Paper-facing validation-only runs may use "train,val" '
        'to keep Test completely unread.'
    ),
)
parser.add_argument(
    '--kd_alpha', type=float, default=0.0,
    help=(
        'Weight for KD loss against teacher soft targets in [0, 1]. '
        '0 = pure supervised (default); 1 = pure KD. '
        'Effective loss = (1 - alpha) * MAE(y, y_pred) + alpha * MAE(y, y_teach).'
    ),
)
parser.add_argument(
    '--temporal_env_risk_strength', type=float, default=0.0,
    help=(
        'Mixing weight for Train-only temporal-environment CVaR on the '
        'supervised+KD loss. Zero disables it.'
    ),
)
parser.add_argument(
    '--temporal_env_count', type=int, default=4,
    help='Number of chronological environments formed within Train.',
)
parser.add_argument(
    '--temporal_env_tail_fraction', type=float, default=0.5,
    help='Upper fraction of present Train environments used by CVaR.',
)
parser.add_argument(
    '--freeze_parent_train_analog_exact_match', type=int, default=0,
    help=(
        'Load the trained rank-2 analog parent and optimize only '
        'analog_exact_match_residual (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_joint_retrieval', type=int, default=0,
    help=(
        'Load a trained retrieval-free parent and jointly optimize only '
        'analog_lowrank_retrieval_residual and '
        'analog_exact_match_residual (0/1)'
    ),
)
parser.add_argument(
    '--frozen_parent_ckpt', type=str, default='none',
    help='Parent state_dict used by a frozen residual-adapter experiment',
)
parser.add_argument(
    '--student_init_ckpt', type=str, default='none',
    help=(
        'Architecture-compatible checkpoint used only to initialize a '
        'low-rank student. It is independent of the KD teacher.'
    ),
)
parser.add_argument(
    '--lowrank_student_finetune', type=int, default=0,
    help=(
        'Initialize and fine-tune a factorized low-rank student from '
        'student_init_ckpt (0/1). This is the teacher-agnostic replacement '
        'for the legacy retrieval_aware_head_finetune name.'
    ),
)
parser.add_argument(
    '--student_train_scope', type=str, default='predictor',
    choices=('readouts', 'predictor', 'all'),
    help='Parameters optimized after low-rank initialization.',
)
parser.add_argument(
    '--student_include_retrieval_adapters', type=int, default=0,
    help=(
        'Also fine-tune the low-rank and exact-history retrieval adapters '
        'when they exist (0/1). Generic second-teacher experiments use 0.'
    ),
)
parser.add_argument(
    '--retrieval_aware_head_finetune', type=int, default=0,
    help=(
        'Load a completed daily-low-rank + exact-history model, freeze the '
        'backbone, and jointly refine only the 12 final predictor readouts '
        'plus the two retrieval adapters (0/1).'
    ),
)
parser.add_argument(
    '--structured_width_prune_finetune', type=int, default=0,
    help=(
        'Initialize a narrower factorized model from frozen_parent_ckpt by '
        'structured hidden-channel pruning, then fine-tune the complete '
        'student model (0/1). Channel importance is computed from the '
        'train-only predictor activation second moment.'
    ),
)
parser.add_argument(
    '--retrieval_aware_full_predictor', type=int, default=0,
    help=(
        'When retrieval-aware fine-tuning is enabled, unfreeze the complete '
        'time_wise_predictor instead of only its final readouts (0/1)'
    ),
)
parser.add_argument('--note', type=str, default='Scope')
parser.add_argument('--load_expid', type=str, default=None)
parser.add_argument(
    '--run_id',
    type=str,
    default=None,
    help=(
        'Optional explicit experiment id for a new training run. Unlike '
        '--load_expid, this still trains and writes a new checkpoint. It is '
        'useful for reproducible multi-stage experiment pipelines.'
    ),
)
parser.add_argument(
    '--parent_lite_head_only', type=int, default=0,
    help=(
        'With retrieval_aware_head_finetune=1, train only the twelve final '
        'prediction readouts and keep all retrieval adapters frozen (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_temporal_decoder', type=int, default=0,
    help=(
        'Load a trained retrieval parent and optimize only the temporal '
        'low-rank decoder (0/1)'
    ),
)
parser.add_argument(
    '--freeze_parent_train_analog_temporal_gap_residual',
    type=int,
    default=0,
    help=(
        'Load a trained retrieval parent and optimize only the final '
        'node-aware temporal gap residual (0/1)'
    ),
)

expid = time.strftime("%m%d%H%M", time.localtime())
args = parser.parse_args()


def get_frozen_adapter_names(config):
    """Return the adapter modules selected for frozen-parent fitting."""
    requested = []
    if bool(getattr(config, 'freeze_parent_train_node_linear', 0)):
        requested.append('node_linear_dynamics')
    if bool(getattr(config, 'freeze_parent_train_regime_residual', 0)):
        requested.append('regime_gated_residual')
    if bool(getattr(
        config, 'freeze_parent_train_multires_block_residual', 0
    )):
        requested.append('multires_block_residual')
    if bool(getattr(config, 'freeze_parent_train_seasonal_drift', 0)):
        requested.append('seasonal_drift_residual')
    if bool(getattr(config, 'freeze_parent_train_analog_retrieval', 0)):
        requested.append('analog_retrieval_residual')
    if bool(getattr(
        config, 'freeze_parent_train_analog_candidate_reranker', 0
    )):
        requested.append('analog_candidate_reranker')
        if bool(getattr(
            config, 'joint_train_lowrank_with_reranker', 0
        )):
            requested.append('analog_lowrank_retrieval_residual')
    if bool(getattr(
        config, 'freeze_parent_train_analog_query_conditioned_residual', 0
    )):
        requested.append('analog_query_conditioned_residual')
    if bool(getattr(
        config, 'freeze_parent_train_analog_correction_utility_gate', 0
    )):
        requested.append('analog_correction_utility_gate')
    if bool(getattr(
        config, 'freeze_parent_train_analog_adaptive_residual_mixer', 0
    )):
        requested.append('analog_adaptive_residual_mixer')
    if bool(getattr(
        config, 'freeze_parent_train_analog_lowrank_retrieval', 0
    )):
        requested.append('analog_lowrank_retrieval_residual')
    if bool(getattr(
        config, 'freeze_parent_train_analog_temporal_decoder', 0
    )):
        requested.append('analog_temporal_lowrank_decoder')
    if bool(getattr(
        config, 'freeze_parent_train_analog_temporal_gap_residual', 0
    )):
        requested.append('analog_temporal_gap_residual')
    if bool(getattr(
        config, 'freeze_parent_train_analog_node_horizon_retrieval', 0
    )):
        requested.append('analog_node_horizon_retrieval_residual')
    if bool(getattr(
        config, 'freeze_parent_train_analog_confidence_aware_retrieval', 0
    )):
        requested.append('analog_confidence_aware_retrieval_residual')
    if bool(getattr(
        config, 'freeze_parent_train_analog_weekly_retrieval', 0
    )):
        requested.append('analog_weekly_retrieval_residual')
    if bool(getattr(
        config, 'freeze_parent_train_analog_metric_retrieval', 0
    )):
        requested.append('analog_metric_retrieval_residual')
    if bool(getattr(
        config, 'freeze_parent_train_analog_exact_match', 0
    )):
        requested.append('analog_exact_match_residual')
    joint_retrieval = bool(getattr(
        config, 'freeze_parent_train_analog_joint_retrieval', 0
    ))
    if joint_retrieval:
        if requested:
            raise ValueError(
                'freeze_parent_train_analog_joint_retrieval cannot be '
                'combined with an individual frozen-adapter flag: '
                f'{requested}'
            )
        return (
            'analog_lowrank_retrieval_residual',
            'analog_exact_match_residual',
        )
    if len(requested) > 3:
        raise ValueError(
            'At most three frozen residual adapters may be trained at a time '
            '(reranker + lowrank + confidence gate, or another combination): '
            f'{requested}'
        )
    return tuple(requested)


def unpack_dataloader_input(loader_input):
    """Return traffic/time and optional train-memory tensors."""
    empty_candidate = (None, None, None)
    if not isinstance(loader_input, (list, tuple)):
        raise TypeError('Dataloader model input must be a list or tuple')
    if len(loader_input) == 2:
        x, x_time = loader_input
        return (x, x_time, None, None, None) + empty_candidate
    if len(loader_input) == 3:
        x, x_time, analog_retrieval = loader_input
        return (
            x, x_time, analog_retrieval, None, None
        ) + empty_candidate
    if len(loader_input) == 5:
        (
            x, x_time, analog_retrieval,
            exact_retrieval, exact_available,
        ) = loader_input
        return (
            x, x_time, analog_retrieval,
            exact_retrieval, exact_available,
        ) + empty_candidate
    if len(loader_input) == 8:
        (
            x, x_time, analog_retrieval,
            exact_retrieval, exact_available,
            candidate_future, candidate_available, candidate_features,
        ) = loader_input
        return (
            x, x_time, analog_retrieval,
            exact_retrieval, exact_available,
            candidate_future, candidate_available, candidate_features,
        )
    raise ValueError(
        f'Expected 2, 3, 5, or 8 dataloader inputs, got {len(loader_input)}'
    )


def get_logger(log_dir, log_filename, name=None):
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    # log_level = 'INFO'
    log_level = 'DEBUG'

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger


class trainer():
    def __init__(self, args, device, scaler, inverse=True):
        self._logger = getLogger()
        self.args = args
        self.scaler = scaler
        self.device = device
        self.clip = 1
        self.eval_mask = args.eval_mask
        self.inverse = inverse
        self.epochs = args.epochs
        self.frozen_adapter_names = get_frozen_adapter_names(args)
        self.lowrank_student_finetune = bool(
            getattr(args, 'lowrank_student_finetune', 0)
            or getattr(args, 'retrieval_aware_head_finetune', 0)
        )
        # Backward-compatible internal alias. Old experiment pickles and
        # configs continue to reproduce their original control flow.
        self.retrieval_aware_head_finetune = self.lowrank_student_finetune
        self.partial_finetune_modules = []

        self.node_tail_risk_scales = None
        node_tail_strength = float(getattr(
            args, 'node_tail_risk_strength', 0.0
        ))
        node_tail_scale_path = str(getattr(
            args, 'node_tail_risk_scale_path', 'none'
        ))
        if node_tail_strength > 0.0 and node_tail_scale_path not in (
            '', 'none'
        ):
            if not os.path.exists(node_tail_scale_path):
                raise FileNotFoundError(
                    'node-tail risk scale file not found: '
                    f'{node_tail_scale_path}'
                )
            with np.load(node_tail_scale_path, allow_pickle=False) as scale_file:
                if 'node_scales' not in scale_file:
                    raise ValueError(
                        f'{node_tail_scale_path} is missing node_scales'
                    )
                self.node_tail_risk_scales = np.asarray(
                    scale_file['node_scales'], dtype=np.float32
                ).reshape(-1)
                stats_dataset = str(np.asarray(
                    scale_file.get('dataset_name', args.dataset_name)
                ).item())
                if stats_dataset != str(args.dataset_name):
                    raise ValueError(
                        'node-tail scale dataset mismatch: '
                        f'{stats_dataset} != {args.dataset_name}'
                    )
            self._logger.info(
                'Loaded Train-only node-tail scales: %s (nodes=%d)',
                node_tail_scale_path,
                self.node_tail_risk_scales.size,
            )

        self.future_change_node_thresholds = None
        future_change_strength = float(getattr(
            args, 'future_change_balance_strength', 0.0
        ))
        if future_change_strength < 0.0:
            raise ValueError(
                'future_change_balance_strength must be non-negative'
            )
        if future_change_strength > 0.0:
            stats_path = str(getattr(
                args, 'future_change_balance_stats_path', 'none'
            ))
            if stats_path in ('', 'none') or not os.path.exists(stats_path):
                raise FileNotFoundError(
                    'future-change balancing requires a Train-only stats '
                    f'file, got {stats_path!r}'
                )
            with np.load(stats_path, allow_pickle=False) as stats_file:
                if 'node_thresholds' not in stats_file:
                    raise ValueError(
                        f'{stats_path} is missing node_thresholds'
                    )
                self.future_change_node_thresholds = np.asarray(
                    stats_file['node_thresholds'], dtype=np.float32
                ).reshape(-1)
                stats_dataset = str(np.asarray(
                    stats_file.get('dataset_name', args.dataset_name)
                ).item())
                if stats_dataset != str(args.dataset_name):
                    raise ValueError(
                        'future-change stats dataset mismatch: '
                        f'{stats_dataset} != {args.dataset_name}'
                    )
            self._logger.info(
                'Loaded Train-only future-change thresholds: %s '
                '(nodes=%d, strength=%.4f)',
                stats_path,
                self.future_change_node_thresholds.size,
                future_change_strength,
            )

        self.change_persistence_history_thresholds = None
        persistence_weight = float(getattr(
            args, 'change_persistence_loss_weight', 0.0
        ))
        if persistence_weight < 0.0:
            raise ValueError(
                'change_persistence_loss_weight must be non-negative'
            )
        if persistence_weight > 0.0:
            stats_path = str(getattr(
                args, 'change_persistence_stats_path', 'none'
            ))
            if stats_path in ('', 'none') or not os.path.exists(stats_path):
                raise FileNotFoundError(
                    'change-persistence supervision requires a Train-only '
                    f'stats file, got {stats_path!r}'
                )
            with np.load(stats_path, allow_pickle=False) as stats_file:
                if 'history_node_thresholds' not in stats_file:
                    raise ValueError(
                        f'{stats_path} is missing history_node_thresholds'
                    )
                self.change_persistence_history_thresholds = np.asarray(
                    stats_file['history_node_thresholds'], dtype=np.float32
                ).reshape(-1)
                stats_dataset = str(np.asarray(
                    stats_file.get('dataset_name', args.dataset_name)
                ).item())
                if stats_dataset != str(args.dataset_name):
                    raise ValueError(
                        'change-persistence stats dataset mismatch: '
                        f'{stats_dataset} != {args.dataset_name}'
                    )
            self._logger.info(
                'Loaded Train-only history-change thresholds: %s '
                '(nodes=%d, loss_weight=%.4f)',
                stats_path,
                self.change_persistence_history_thresholds.size,
                persistence_weight,
            )

        self.change_event_history_thresholds = None
        event_weight = float(getattr(
            args, 'change_event_loss_weight', 0.0
        ))
        event_emphasis = float(getattr(
            args, 'change_event_emphasis', 2.0
        ))
        if event_weight < 0.0:
            raise ValueError('change_event_loss_weight must be non-negative')
        if event_emphasis < 0.0:
            raise ValueError('change_event_emphasis must be non-negative')
        if event_weight > 0.0:
            stats_path = str(getattr(
                args, 'change_event_stats_path', 'none'
            ))
            if stats_path in ('', 'none') or not os.path.exists(stats_path):
                raise FileNotFoundError(
                    'signed-event supervision requires a Train-only stats '
                    f'file, got {stats_path!r}'
                )
            with np.load(stats_path, allow_pickle=False) as stats_file:
                if 'history_node_thresholds' not in stats_file:
                    raise ValueError(
                        f'{stats_path} is missing history_node_thresholds'
                    )
                self.change_event_history_thresholds = np.asarray(
                    stats_file['history_node_thresholds'], dtype=np.float32
                ).reshape(-1)
                stats_dataset = str(np.asarray(
                    stats_file.get('dataset_name', args.dataset_name)
                ).item())
                if stats_dataset != str(args.dataset_name):
                    raise ValueError(
                        'signed-event stats dataset mismatch: '
                        f'{stats_dataset} != {args.dataset_name}'
                    )
            self._logger.info(
                'Loaded Train-only history thresholds for signed events: %s '
                '(nodes=%d, loss_weight=%.4f, event_emphasis=%.4f)',
                stats_path,
                self.change_event_history_thresholds.size,
                event_weight,
                event_emphasis,
            )

        self.model = self._build_model(args.modelid)
        self.loss = self._build_loss_func(args.loss_type)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.lr_scheduler = self._build_lr_scheduler()

    def _build_model(self, modelid):
        model_dict = {"TLAST": TLAST, "TLAST_improved": Scope,
                      "Scope": Scope}
        if modelid in model_dict.keys():
            model = model_dict[modelid].Model(self.args) 
        else:
            self._logger.error(
                f'no model named `{modelid}`, please try again!')
            sys.exit('error')

        preserved_parameters = set()
        for module_name, module in model.named_modules():
            if getattr(module, 'preserve_initialization', False):
                for parameter_name, _ in module.named_parameters(recurse=True):
                    full_name = '.'.join(
                        part for part in (module_name, parameter_name) if part
                    )
                    preserved_parameters.add(full_name)

        for name, p in model.named_parameters():
            if p.dim() > 1 and name not in preserved_parameters:
                nn.init.xavier_uniform_(p)
        if preserved_parameters:
            self._logger.info(
                f'Preserved explicit initialization for: '
                f'{sorted(preserved_parameters)}'
            )
        model.to(self.device)

        self._logger.info(model)
        for name, param in model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_para_num, trainable_para_num = get_parameter_number(model)
        self._logger.info(f'total_parameter_num is {total_para_num}, trainable_parameter_num is {trainable_para_num}')   
        return model
    
    def _build_loss_func(self, loss_type):
        if loss_type == 'huber':
            loss_func = partial(utils.huber_loss, delta=self.args.huber_delta)
        elif loss_type == 'masked_huber_0':
            loss_func = partial(utils.masked_huber_loss, delta=self.args.huber_delta, null_val = 0.)
        elif loss_type == 'masked_mae_0':
            loss_func = partial(utils.masked_mae_torch, null_val = 0.)
        elif loss_type == 'masked_huber_eval':
            loss_func = partial(utils.masked_huber_loss, delta=self.args.huber_delta, null_val = 0., mask_val= self.args.eval_mask)
        else:
            raise ValueError(f'Unsupported loss_type: {loss_type}')
        return loss_func
    
    def _build_lr_scheduler(self):
        self.lr_decay = bool(self.args.lr_decay)
        if self.lr_decay:            
            self.lr_scheduler_type = self.args.lr_scheduler_type
            self.lr_decay_ratio = 0.1
            self.lr_T_max = 30
            self.lr_eta_min = 0
            self.lr_warmup_epoch = 5
            self.lr_warmup_init = 1e-6
            if self.lr_scheduler_type.lower() == 'cosinelr':
                lr_scheduler = utils.CosineLRScheduler(
                    self.optimizer, t_initial=self.epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                    warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init)
                self._logger.info(f'Use {self.lr_scheduler_type.lower()} lr_scheduler.')
            else:
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def _forward(self, input):
        output, _ = self.model(input)
        if self.inverse:
            predict = self.scaler.inverse_transform(output)
        else:
            predict = output
        
        return predict

    def _metric(self, predict, real):
        if self.eval_mask == -1:
            mae = utils.mae(predict, real).item()
            rmse = utils.rmse(predict, real).item()
            mape = utils.masked_mape_torch(predict, real, 0.0).item()
        elif self.eval_mask == 0:
            mae = utils.masked_mae_torch(predict, real, 0.0).item()
            rmse = utils.masked_rmse_torch(predict, real, 0.0).item()
            mape = utils.masked_mape_torch(predict, real, 0.0).item()
        elif self.eval_mask > 0:
            mae = utils.masked_mae_torch(
                predict, real, 0.0, self.eval_mask).item()
            rmse = utils.masked_rmse_torch(
                predict, real, 0.0, self.eval_mask).item()
            mape = utils.masked_mape_torch(
                predict, real, 0.0, self.eval_mask).item()
        return mae, mape, rmse

    def _aggregation_counts(self, real_val):
        """Counts needed to combine batch metrics into exact dataset metrics."""
        loss_type = self.args.loss_type
        if loss_type == 'huber':
            loss_count = real_val.numel()
        elif loss_type == 'masked_huber_eval':
            loss_count = utils.masked_valid_count_torch(
                real_val, 0.0, self.eval_mask
            ).item()
        else:
            loss_count = utils.masked_valid_count_torch(real_val, 0.0).item()

        if self.eval_mask == -1:
            metric_count = real_val.numel()
            mape_count = utils.masked_valid_count_torch(real_val, 0.0).item()
        elif self.eval_mask == 0:
            metric_count = utils.masked_valid_count_torch(real_val, 0.0).item()
            mape_count = metric_count
        else:
            metric_count = utils.masked_valid_count_torch(
                real_val, 0.0, self.eval_mask
            ).item()
            mape_count = metric_count

        return {
            'loss': float(loss_count),
            'mae': float(metric_count),
            'mape': float(mape_count),
            'rmse': float(metric_count),
        }

    def train(
        self, input, real_val, batches_seen, teacher_pred=None,
        temporal_position=None,
    ):
        """
        训练一个 batch

        修改点：
            1. 从 self.model.ortho_loss 获取正交损失
            2. 总损失 = main_loss + lambda_ortho * ortho_loss
            3. 反向传播使用总损失
            4. KD mixin: when teacher_pred is not None, blend the per-element
               MAE against the soft target with weight kd_alpha.
        """
        if self.frozen_adapter_names:
            # The parent checkpoint must remain a deterministic fixed
            # predictor while the local residual branch is fitted.  Calling
            # model.train() here would reactivate all parent dropouts even
            # though their parameters are frozen, changing the residual target
            # on every batch.
            self.model.eval()
            for adapter_name in self.frozen_adapter_names:
                getattr(self.model, adapter_name).train()
        elif self.retrieval_aware_head_finetune:
            # The frozen representation must remain deterministic.  Only the
            # final horizon readouts and the two already-trained train-memory
            # adapters are re-optimized, so leave every backbone dropout in
            # eval mode and reactivate exactly those trainable modules.
            self.model.eval()
            for module in self.partial_finetune_modules:
                module.train()
        else:
            self.model.train()
        self.optimizer.zero_grad()
        
        # ===== 前向传播 =====
        # KD soft targets, when present, are passed in via the teacher_pred
        # keyword argument (see train loop). The model's forward signature is
        # untouched — we never fold teacher pred into the input list, so the
        # 2/3/4/6/9-input contract used elsewhere still holds.
        predict = self._forward(input)
        volatility_strength = float(
            getattr(self.args, 'volatility_mae_strength', 0.0)
        )
        volatility_quantile = float(
            getattr(self.args, 'volatility_mae_quantile', 0.75)
        )
        if volatility_strength < 0.0:
            raise ValueError('volatility_mae_strength must be non-negative')
        if not 0.0 < volatility_quantile < 1.0:
            raise ValueError(
                'volatility_mae_quantile must be in the interval (0, 1)'
            )
        volatility_stats = None
        future_change_stats = None
        node_tail_risk_stats = None

        # ===== Train-label future-change balanced MAE (optional) =====
        # This is supervised training, not an inference gate: labels define
        # the high-change group only while optimizing Train.  The thresholds
        # are fixed per node from the complete Train split, and evaluation is
        # always the unweighted official metric.
        future_change_strength = float(getattr(
            self.args, 'future_change_balance_strength', 0.0
        ))
        if future_change_strength > 0.0:
            if getattr(self.args, 'time_weights', None) is not None:
                raise ValueError(
                    'future-change balancing cannot be combined with '
                    'time_weights'
                )
            if volatility_strength > 0.0:
                raise ValueError(
                    'future-change balancing cannot be combined with '
                    'history-volatility weighting'
                )
            history_input = input[0] if isinstance(input, (list, tuple)) else input
            with torch.no_grad():
                history_raw = self.scaler.inverse_transform(
                    history_input.detach()
                )
                future_change_weights, future_change_stats = (
                    utils.future_change_balance_weights_torch(
                        history_raw,
                        real_val,
                        self.future_change_node_thresholds,
                        weight_strength=future_change_strength,
                        null_val=0.0,
                    )
                )
            main_loss = utils.masked_weighted_mae_torch(
                predict,
                real_val,
                future_change_weights,
                null_val=0.0,
            )

        # ===== 历史波动加权 MAE（可选，单变量实验）=====
        # 权重只读取输入12步。先还原到原始流量尺度，保证真实零值可以
        # 作为缺失端点排除；验证/Test仍通过 self.loss 使用普通masked MAE。
        elif volatility_strength > 0.0:
            if getattr(self.args, 'time_weights', None) is not None:
                raise ValueError(
                    'volatility-weighted MAE cannot be combined with time_weights'
                )
            history_input = input[0] if isinstance(input, (list, tuple)) else input
            with torch.no_grad():
                history_raw = self.scaler.inverse_transform(
                    history_input.detach()
                )
                volatility_weights, volatility_stats = (
                    utils.history_volatility_weights_torch(
                        history_raw,
                        high_quantile=volatility_quantile,
                        weight_strength=volatility_strength,
                        null_val=0.0,
                    )
                )
            main_loss = utils.masked_weighted_mae_torch(
                predict,
                real_val,
                volatility_weights,
                null_val=0.0,
            )

        # ===== 时间步加权损失（可选，单变量 ablation）=====
        # 若设置了 time_weights（逗号分隔的 T 个权重），对 12 个预测步施加非均匀权重，
        # 迫使模型优先拟合短期（易）步，从而拉低平均 MAE。评估/报告仍用标准无加权 MAE。
        elif getattr(self.args, 'time_weights', None) is not None:
            tw = [float(w) for w in str(self.args.time_weights).split(',')]
            tw = torch.tensor(tw, device=predict.device, dtype=predict.dtype)  # [T]
            T = predict.shape[1]
            if len(tw) != T:
                self._logger.warning(f"time_weights 长度 {len(tw)} 与预测步数 {T} 不符，已截断/补1")
                if len(tw) > T:
                    tw = tw[:T]
                else:
                    tw = torch.cat([tw, torch.ones(T - len(tw), device=predict.device, dtype=predict.dtype)])
            tw = tw / tw.mean().clamp_min(torch.finfo(tw.dtype).eps)
            per_step = torch.stack([
                utils.masked_mae_torch(
                    predict[:, horizon], real_val[:, horizon], 0.0
                )
                for horizon in range(T)
            ])
            main_loss = (per_step * tw).mean()
        else:
            main_loss = self.loss(predict, real_val)

        # ===== Node-level spatial tail-risk objective (optional) =====
        # The upper-risk set is recomputed from each Train batch, so no
        # Validation/Test node identity or target is used to select difficult
        # sensors.  This changes only optimization; inference and the official
        # validation metric remain exactly the same as the baseline.
        node_tail_risk_strength = float(getattr(
            self.args, 'node_tail_risk_strength', 0.0
        ))
        node_tail_risk_fraction = float(getattr(
            self.args, 'node_tail_risk_fraction', 0.1
        ))
        if not 0.0 <= node_tail_risk_strength <= 1.0:
            raise ValueError(
                'node_tail_risk_strength must be in the interval [0, 1]'
            )
        if not 0.0 < node_tail_risk_fraction <= 1.0:
            raise ValueError(
                'node_tail_risk_fraction must be in the interval (0, 1]'
            )
        if node_tail_risk_strength > 0.0:
            if getattr(self.args, 'time_weights', None) is not None:
                raise ValueError(
                    'node tail risk cannot be combined with time_weights'
                )
            if volatility_strength > 0.0 or future_change_strength > 0.0:
                raise ValueError(
                    'node tail risk must be tested separately from existing '
                    'sample reweighting objectives'
                )
            node_tail_loss, node_tail_risk_stats = (
                utils.masked_node_cvar_mae_torch(
                    predict,
                    real_val,
                    tail_fraction=node_tail_risk_fraction,
                    node_scales=self.node_tail_risk_scales,
                    null_val=0.0,
                )
            )
            main_loss = (
                (1.0 - node_tail_risk_strength) * main_loss
                + node_tail_risk_strength * node_tail_loss
            )

        # ===== Knowledge distillation loss (optional) =====
        # Soft target = precomputed teacher forward over the same window,
        # already in the same z-score space. KD mixin lets the student
        # inherit nonparametric corrections without needing the cache.
        kd_alpha = float(getattr(self.args, 'kd_alpha', 0.0))
        kd_loss = None
        if kd_alpha > 0.0 and teacher_pred is None:
            raise RuntimeError(
                'kd_alpha > 0 requires an aligned teacher prediction for '
                'every training batch'
            )
        # Dataset loaders expose the aligned teacher batch as a NumPy array.
        # Convert it here instead of silently skipping KD because the object is
        # not already a torch.Tensor.
        if kd_alpha > 0.0 and teacher_pred is not None:
            teacher_pred = torch.as_tensor(
                teacher_pred,
                device=predict.device,
                dtype=torch.float32,
            )
            teacher_space = str(getattr(
                self.args, 'teacher_predictions_space', 'normalized'
            )).lower()
            if teacher_space not in ('normalized', 'raw'):
                raise ValueError(
                    'teacher_predictions_space must be normalized or raw, '
                    f'got {teacher_space!r}'
                )
            # Match the soft target to the active supervised-loss space.
            if self.inverse and teacher_space == 'normalized':
                teacher_pred = self.scaler.inverse_transform(teacher_pred)
            elif not self.inverse and teacher_space == 'raw':
                teacher_pred = self.scaler.transform(teacher_pred)
        if (
            kd_alpha > 0.0
            and teacher_pred is not None
            and isinstance(teacher_pred, torch.Tensor)
        ):
            if teacher_pred.shape != predict.shape:
                raise ValueError(
                    'KD soft target shape does not match the student output: '
                    f'{tuple(teacher_pred.shape)} != {tuple(predict.shape)}'
                )
            kd_loss = utils.masked_mae_torch(
                predict, teacher_pred, 0.0
            )
            main_loss = (
                (1.0 - kd_alpha) * main_loss
                + kd_alpha * kd_loss
            )

        # ===== Horizon-Ordered Gradient Balancing (training only) =====
        # The twelve forecasting horizons share a representation but their
        # gradient norms grow systematically with forecasting distance.  At a
        # sparse update interval, estimate each combined supervised+KD loss's
        # gradient on that shared representation, smooth the profile along the
        # ordered horizon axis, and inversely reweight it.  Detached, clipped
        # weights keep this a first-order optimizer modification; inference is
        # unchanged and gains no parameters.
        horizon_balance_beta = float(getattr(
            self.args, 'horizon_grad_balance_beta', 0.0
        ))
        horizon_balance_stats = None
        if horizon_balance_beta < 0.0:
            raise ValueError('horizon_grad_balance_beta must be non-negative')
        if horizon_balance_beta > 0.0:
            if getattr(self.args, 'time_weights', None) is not None:
                raise ValueError(
                    'dynamic horizon gradient balancing cannot be combined '
                    'with static time_weights'
                )
            if (
                volatility_strength > 0.0
                or future_change_strength > 0.0
                or node_tail_risk_strength > 0.0
            ):
                raise ValueError(
                    'dynamic horizon gradient balancing must be tested '
                    'separately from sample/node reweighting objectives'
                )
            momentum = float(getattr(
                self.args, 'horizon_grad_balance_momentum', 0.9
            ))
            interval = int(getattr(
                self.args, 'horizon_grad_balance_interval', 50
            ))
            warmup_steps = int(getattr(
                self.args, 'horizon_grad_balance_warmup_steps', 100
            ))
            min_weight = float(getattr(
                self.args, 'horizon_grad_balance_min_weight', 0.5
            ))
            max_weight = float(getattr(
                self.args, 'horizon_grad_balance_max_weight', 1.5
            ))
            if not 0.0 <= momentum < 1.0:
                raise ValueError(
                    'horizon_grad_balance_momentum must be in [0, 1)'
                )
            if interval <= 0 or warmup_steps < 0:
                raise ValueError(
                    'horizon gradient balance interval must be positive and '
                    'warmup_steps must be non-negative'
                )
            if not 0.0 < min_weight <= max_weight:
                raise ValueError('invalid horizon gradient weight bounds')
            predictor = getattr(self.model, 'time_wise_predictor', None)
            shared_hidden = getattr(
                predictor, 'last_shared_hidden', None
            )
            if shared_hidden is None or not shared_hidden.requires_grad:
                raise RuntimeError(
                    'horizon gradient balancing requires a differentiable '
                    'time_wise_predictor.last_shared_hidden tensor'
                )
            horizon_count = int(predict.shape[1])
            hard_horizon_losses = torch.stack([
                utils.masked_mae_torch(
                    predict[:, horizon], real_val[:, horizon], 0.0
                )
                for horizon in range(horizon_count)
            ])
            if kd_alpha > 0.0 and teacher_pred is not None:
                kd_horizon_losses = torch.stack([
                    utils.masked_mae_torch(
                        predict[:, horizon],
                        teacher_pred[:, horizon],
                        0.0,
                    )
                    for horizon in range(horizon_count)
                ])
                combined_horizon_losses = (
                    (1.0 - kd_alpha) * hard_horizon_losses
                    + kd_alpha * kd_horizon_losses
                )
            else:
                combined_horizon_losses = hard_horizon_losses

            should_update_profile = (
                batches_seen >= warmup_steps
                and (batches_seen - warmup_steps) % interval == 0
            )
            ema_norm = getattr(self, '_horizon_grad_norm_ema', None)
            if should_update_profile or ema_norm is None:
                current_norms = []
                for horizon_loss in combined_horizon_losses:
                    hidden_gradient = torch.autograd.grad(
                        horizon_loss,
                        shared_hidden,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=False,
                    )[0]
                    current_norms.append(
                        hidden_gradient.detach().float().norm(p=2)
                    )
                current_norms = torch.stack(current_norms)
                if ema_norm is None:
                    ema_norm = current_norms
                else:
                    ema_norm = (
                        momentum * ema_norm.to(current_norms.device)
                        + (1.0 - momentum) * current_norms
                    )
                self._horizon_grad_norm_ema = ema_norm.detach()

            ordered_norm = ema_norm.to(
                device=predict.device, dtype=predict.dtype
            ).clone()
            if horizon_count > 1:
                smoothed_norm = ordered_norm.clone()
                smoothed_norm[0] = (
                    0.75 * ordered_norm[0] + 0.25 * ordered_norm[1]
                )
                smoothed_norm[-1] = (
                    0.25 * ordered_norm[-2] + 0.75 * ordered_norm[-1]
                )
                if horizon_count > 2:
                    smoothed_norm[1:-1] = (
                        0.25 * ordered_norm[:-2]
                        + 0.50 * ordered_norm[1:-1]
                        + 0.25 * ordered_norm[2:]
                    )
            else:
                smoothed_norm = ordered_norm
            eps = torch.finfo(predict.dtype).eps
            if batches_seen < warmup_steps:
                balance_weights = torch.ones_like(smoothed_norm)
            else:
                balance_weights = (
                    smoothed_norm.mean().clamp_min(eps)
                    / smoothed_norm.clamp_min(eps)
                ).pow(horizon_balance_beta)
                balance_weights = balance_weights.clamp(
                    min=min_weight, max=max_weight
                )
                balance_weights = (
                    balance_weights
                    / balance_weights.mean().clamp_min(eps)
                )
            balance_weights = balance_weights.detach()
            main_loss = (
                combined_horizon_losses * balance_weights
            ).mean()
            horizon_balance_stats = {
                'weights': balance_weights,
                'gradient_norms': ordered_norm.detach(),
            }
            if batches_seen % 100 == 0:
                self._logger.info(
                    '[HOGB] step=%d norms=%s weights=%s',
                    batches_seen,
                    [round(value, 6) for value in ordered_norm.tolist()],
                    [round(value, 6) for value in balance_weights.tolist()],
                )

        # ===== Train-only temporal-environment robust distillation =====
        # The loader keeps each shuffled sample's normalized position in the
        # Train split. CVaR therefore targets the worst chronological Train
        # environments without using Validation/Test to define groups.
        temporal_env_strength = float(getattr(
            self.args, 'temporal_env_risk_strength', 0.0
        ))
        temporal_env_stats = None
        if not 0.0 <= temporal_env_strength <= 1.0:
            raise ValueError('temporal_env_risk_strength must be in [0, 1]')
        if temporal_env_strength > 0.0:
            if teacher_pred is None or kd_alpha <= 0.0:
                raise ValueError(
                    'temporal environment risk requires KD soft targets'
                )
            if temporal_position is None:
                raise ValueError(
                    'temporal environment risk requires Train positions'
                )
            if node_tail_risk_strength > 0.0:
                raise ValueError(
                    'temporal environment risk and node tail risk must be '
                    'tested separately'
                )
            temporal_tail_loss, temporal_env_stats = (
                utils.masked_temporal_environment_cvar_torch(
                    predict,
                    real_val,
                    teacher_pred,
                    temporal_position,
                    kd_alpha=kd_alpha,
                    num_environments=int(getattr(
                        self.args, 'temporal_env_count', 4
                    )),
                    tail_fraction=float(getattr(
                        self.args, 'temporal_env_tail_fraction', 0.5
                    )),
                    null_val=0.0,
                )
            )
            main_loss = (
                (1.0 - temporal_env_strength) * main_loss
                + temporal_env_strength * temporal_tail_loss
            )

        # ===== 相邻 horizon 差分一致性（可选，单变量实验）=====
        # 标准 MAE 仍是主目标；该项仅约束未来曲线的相邻变化量。
        # 掩码依据原始标签的两个端点，不能把合法的零差分当成缺失值。
        diff_weight = float(
            getattr(self.args, 'diff_consistency_weight', 0.0)
        )
        if diff_weight < 0:
            raise ValueError('diff_consistency_weight must be non-negative')
        diff_loss = None
        total_loss = main_loss

        # One-checkpoint nested-rank Student. The supervised/KD objective
        # trains the discrete routed prediction; these two Train-only terms
        # keep the average budget on target and preserve the rank-64 subnet.
        elastic_budget_loss = None
        elastic_consistency_loss = None
        if bool(getattr(self.args, 'use_elastic_rank_router', 0)):
            predictor = self.model.time_wise_predictor
            probability = getattr(
                predictor, 'last_elastic_probability', None
            )
            low_output = getattr(
                predictor, 'last_elastic_low_output', None
            )
            high_output = getattr(
                predictor, 'last_elastic_high_output', None
            )
            if probability is None or low_output is None or high_output is None:
                raise RuntimeError('elastic rank state was not produced')
            target_fraction = float(getattr(
                self.args, 'elastic_target_high_fraction', 0.5
            ))
            if not 0.0 <= target_fraction <= 1.0:
                raise ValueError('elastic_target_high_fraction must be in [0,1]')
            elastic_budget_loss = (
                probability.mean() - target_fraction
            ).pow(2)
            elastic_consistency_loss = torch.nn.functional.l1_loss(
                low_output, high_output.detach()
            )
            total_loss = total_loss + float(getattr(
                self.args, 'elastic_budget_loss_weight', 1.0
            )) * elastic_budget_loss
            total_loss = total_loss + float(getattr(
                self.args, 'elastic_low_consistency_weight', 0.2
            )) * elastic_consistency_loss
            if batches_seen % 100 == 0:
                hard_fraction = getattr(
                    predictor, 'last_elastic_hard_gate'
                ).float().mean()
                self._logger.info(
                    '[ElasticRank] step=%d probability=%.4f hard_high=%.4f '
                    'budget_loss=%.6f consistency=%.6f',
                    batches_seen, probability.mean().item(),
                    hard_fraction.item(), elastic_budget_loss.item(),
                    elastic_consistency_loss.item(),
                )

        # ===== Single-Student reliability supervision (optional) =====
        # The point forecast is detached before entering the risk head inside
        # the model, so this auxiliary objective cannot perturb the predictor.
        # Targets are per-window Train MAE in log space; no Validation/Test
        # target participates in fitting the confidence score.
        risk_weight = float(getattr(
            self.args, 'student_risk_loss_weight', 0.0
        ))
        if risk_weight < 0.0:
            raise ValueError('student_risk_loss_weight must be non-negative')
        risk_loss = None
        if risk_weight > 0.0:
            risk_score = getattr(self.model, 'last_risk_score', None)
            if risk_score is None:
                raise ValueError(
                    'student_risk_loss_weight > 0 requires '
                    'use_student_risk_head=1'
                )
            valid = torch.isfinite(real_val) & real_val.ne(0.0)
            valid_float = valid.to(predict.dtype)
            absolute_error = torch.where(
                valid,
                torch.abs(predict.detach() - real_val),
                torch.zeros_like(predict),
            )
            reduce_dims = tuple(range(1, absolute_error.ndim))
            sample_mae = (
                absolute_error.sum(dim=reduce_dims)
                / valid_float.sum(dim=reduce_dims).clamp_min(1.0)
            )
            risk_target = torch.log1p(sample_mae).detach()
            risk_loss = torch.nn.functional.smooth_l1_loss(
                risk_score, risk_target
            )
            total_loss = total_loss + risk_weight * risk_loss

        # ===== Active-history change-persistence supervision (optional) =====
        # Only Train windows whose observed history is above a fixed per-node
        # Train threshold receive this trajectory-statistic loss.  It teaches
        # the predictor to distinguish persistent change from reversion while
        # leaving the inference graph and official Validation metric intact.
        persistence_weight = float(getattr(
            self.args, 'change_persistence_loss_weight', 0.0
        ))
        if persistence_weight < 0.0:
            raise ValueError(
                'change_persistence_loss_weight must be non-negative'
            )
        persistence_loss = None
        persistence_stats = None
        if persistence_weight > 0.0:
            history_input = (
                input[0] if isinstance(input, (list, tuple)) else input
            )
            history_raw = self.scaler.inverse_transform(history_input)
            persistence_loss, persistence_stats = (
                utils.active_history_change_persistence_loss_torch(
                    history_raw,
                    predict,
                    real_val,
                    self.change_persistence_history_thresholds,
                    null_val=0.0,
                )
            )
            total_loss = total_loss + persistence_weight * persistence_loss

        # ===== Signed per-horizon change-event supervision (optional) =====
        # Unlike total-variation matching, this retains the sign and exact
        # horizon of every transition.  The mask is derived only from the
        # observed input; future labels are ordinary Train supervision.
        event_weight = float(getattr(
            self.args, 'change_event_loss_weight', 0.0
        ))
        event_emphasis = float(getattr(
            self.args, 'change_event_emphasis', 2.0
        ))
        if event_weight < 0.0:
            raise ValueError('change_event_loss_weight must be non-negative')
        if event_emphasis < 0.0:
            raise ValueError('change_event_emphasis must be non-negative')
        event_loss = None
        event_stats = None
        if event_weight > 0.0:
            history_input = (
                input[0] if isinstance(input, (list, tuple)) else input
            )
            history_raw = self.scaler.inverse_transform(history_input)
            event_loss, event_stats = (
                utils.active_history_signed_event_loss_torch(
                    history_raw,
                    predict,
                    real_val,
                    self.change_event_history_thresholds,
                    event_emphasis=event_emphasis,
                    null_val=0.0,
                )
            )
            total_loss = total_loss + event_weight * event_loss
        if diff_weight > 0:
            diff_loss = utils.masked_adjacent_difference_mae_torch(
                predict, real_val, null_val=0.0
            )
            total_loss = total_loss + diff_weight * diff_loss

        # ===== Multi-resolution future-block auxiliary MAE =====
        # This supervises low-frequency 3-step block means during training.
        # It never replaces the standard validation/Test metric and uses the
        # same future labels already authorized for supervised training.
        block_weight = float(
            getattr(self.args, 'multires_block_loss_weight', 0.0)
        )
        if block_weight < 0:
            raise ValueError(
                'multires_block_loss_weight must be non-negative'
            )
        block_loss = None
        if block_weight > 0:
            if not hasattr(self.model, 'multires_block_residual'):
                raise ValueError(
                    'multires_block_loss_weight > 0 requires '
                    'use_multires_block_residual=1'
                )
            block_loss = utils.masked_block_mean_mae_torch(
                predict,
                real_val,
                block_size=self.model.multires_block_residual.block_size,
                null_val=0.0,
            )
            total_loss = total_loss + block_weight * block_loss

        # ===== Train-only candidate utility ranking =====
        # The reranker remains causal at inference.  Training targets are
        # used only here to supervise which of the already selected K
        # train-memory continuations is least harmful / most useful relative
        # to the frozen pre-rerank parent forecast.
        utility_weight = float(
            getattr(self.args, 'analog_utility_loss_weight', 0.0)
        )
        utility_temperature = float(
            getattr(self.args, 'analog_utility_temperature', 1.0)
        )
        if utility_weight < 0.0:
            raise ValueError(
                'analog_utility_loss_weight must be non-negative'
            )
        if utility_temperature <= 0.0:
            raise ValueError(
                'analog_utility_temperature must be positive'
            )
        utility_loss = None
        utility_stats = None
        if utility_weight > 0.0:
            if not hasattr(self.model, 'analog_candidate_reranker'):
                raise ValueError(
                    'analog_utility_loss_weight > 0 requires '
                    'use_analog_candidate_reranker=1'
                )
            target_normalized = self.scaler.transform(real_val)
            target_valid = (
                torch.isfinite(real_val)
                & (torch.abs(real_val) >= 1e-4)
            )
            utility_loss, utility_stats = (
                self.model.analog_candidate_reranker.utility_ranking_loss(
                    target_normalized,
                    target_valid,
                    temperature=utility_temperature,
                )
            )
            total_loss = total_loss + utility_weight * utility_loss

        # ===== 添加正交损失（如果使用了 MAPA）=====
        if hasattr(self.model, 'ortho_loss') and self.model.ortho_loss is not None:
            ortho_loss = self.model.ortho_loss
            if isinstance(ortho_loss, torch.Tensor):
                # 确保 lambda_ortho 参数存在（默认 0.01）
                lambda_ortho = getattr(self.args, 'lambda_ortho', 0.01)
                total_loss = total_loss + lambda_ortho * ortho_loss
                # 调试信息（每100个batch打印一次）
                if batches_seen % 100 == 0:
                    diff_text = (
                        f', diff_loss={diff_loss.item():.4f}, '
                        f'diff_weight={diff_weight:.4f}'
                        if diff_loss is not None else ''
                    )
                    block_text = (
                        f', block_loss={block_loss.item():.4f}, '
                        f'block_weight={block_weight:.4f}'
                        if block_loss is not None else ''
                    )
                    volatility_text = (
                        ', volatility_threshold='
                        f'{volatility_stats["threshold"].item():.4f}, '
                        'volatility_high_fraction='
                        f'{volatility_stats["high_fraction"].item():.4f}, '
                        f'volatility_strength={volatility_strength:.4f}'
                        if volatility_stats is not None else ''
                    )
                    future_change_text = (
                        ', future_change_high_fraction='
                        f'{future_change_stats["high_fraction"].item():.4f}, '
                        'future_change_mean_score='
                        f'{future_change_stats["mean_score"].item():.4f}, '
                        'future_change_strength='
                        f'{future_change_strength:.4f}'
                        if future_change_stats is not None else ''
                    )
                    persistence_text = (
                        f', persistence_loss={persistence_loss.item():.4f}, '
                        f'persistence_weight={persistence_weight:.4f}, '
                        'persistence_active_fraction='
                        f'{persistence_stats["active_fraction"].item():.4f}, '
                        'persistence_true_change='
                        f'{persistence_stats["true_change"].item():.4f}, '
                        'persistence_predicted_change='
                        f'{persistence_stats["predicted_change"].item():.4f}'
                        if persistence_loss is not None else ''
                    )
                    event_text = (
                        f', event_loss={event_loss.item():.4f}, '
                        f'event_weight={event_weight:.4f}, '
                        'event_active_fraction='
                        f'{event_stats["active_fraction"].item():.4f}, '
                        'event_significant_fraction='
                        f'{event_stats["significant_transition_fraction"].item():.4f}, '
                        'event_direction_accuracy='
                        f'{event_stats["significant_direction_accuracy"].item():.4f}, '
                        'event_true_abs_change='
                        f'{event_stats["true_abs_change"].item():.4f}, '
                        'event_predicted_abs_change='
                        f'{event_stats["predicted_abs_change"].item():.4f}'
                        if event_loss is not None else ''
                    )
                    kd_text = (
                        f', kd_loss={kd_loss.item():.4f}, '
                        f'kd_alpha={kd_alpha:.2f}'
                        if kd_loss is not None else ''
                    )
                    node_tail_text = (
                        ', node_tail_loss='
                        f'{node_tail_risk_stats["tail_loss"].item():.4f}, '
                        'node_mean_loss='
                        f'{node_tail_risk_stats["mean_node_loss"].item():.4f}, '
                        'node_tail_count='
                        f'{int(node_tail_risk_stats["tail_count"].item())}, '
                        'node_tail_normalized_risk='
                        f'{node_tail_risk_stats["tail_normalized_risk"].item():.6f}, '
                        'node_tail_strength='
                        f'{node_tail_risk_strength:.4f}'
                        if node_tail_risk_stats is not None else ''
                    )
                    utility_text = (
                        f', utility_loss={utility_loss.item():.4f}, '
                        f'utility_weight={utility_weight:.4f}, '
                        'utility_valid_positions='
                        f'{utility_stats["valid_positions"]}, '
                        'utility_target_best_probability='
                        f'{utility_stats["target_best_probability"]:.4f}'
                        if utility_loss is not None else ''
                    )
                    print(
                        f"[Debug] main_loss={main_loss.item():.4f}"
                        f"{diff_text}{block_text}{volatility_text}"
                        f"{future_change_text}{persistence_text}{event_text}"
                        f"{kd_text}{node_tail_text}{utility_text}, "
                        f"ortho_loss={ortho_loss.item():.4f}, "
                        f"total_loss={total_loss.item():.4f}"
                    )
        
        # ===== 反向传播（使用总损失）=====
        total_loss.backward()
        
        # ===== 梯度裁剪 + 优化器步进 =====
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            if self.lr_scheduler_type.lower() == 'cosinelr':
                self.lr_scheduler.step_update(num_updates=batches_seen)
        
        # ===== 计算评估指标 =====
        mae, mape, rmse = self._metric(predict, real_val)
        return total_loss.item(), mae, mape, rmse

    @torch.no_grad()
    def eval(self, input, real_val, return_counts=False):
        self.model.eval()
        predict = self._forward(input)
        loss = self.loss(predict, real_val)
        mae, mape, rmse = self._metric(predict, real_val)
        if return_counts:
            counts = self._aggregation_counts(real_val)
            horizon_values = {
                'loss': [], 'mae': [], 'mape': [], 'rmse': [],
                'loss_count': [], 'mae_count': [],
                'mape_count': [], 'rmse_count': [],
            }
            for horizon in range(real_val.shape[1]):
                horizon_predict = predict[:, horizon]
                horizon_real = real_val[:, horizon]
                horizon_loss = self.loss(horizon_predict, horizon_real).item()
                horizon_mae, horizon_mape, horizon_rmse = self._metric(
                    horizon_predict, horizon_real
                )
                horizon_counts = self._aggregation_counts(horizon_real)
                horizon_values['loss'].append(horizon_loss)
                horizon_values['mae'].append(horizon_mae)
                horizon_values['mape'].append(horizon_mape)
                horizon_values['rmse'].append(horizon_rmse)
                for metric_name in ('loss', 'mae', 'mape', 'rmse'):
                    horizon_values[f'{metric_name}_count'].append(
                        horizon_counts[metric_name]
                    )
            counts['horizon'] = horizon_values
            return loss.item(), mae, mape, rmse, counts
        return loss.item(), mae, mape, rmse

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num


def _topk_sorted_indices(scores, keep, name):
    """Return deterministic top-k indices in source order."""
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    if not 0 < keep <= scores.numel():
        raise ValueError(
            f'Invalid {name} keep count {keep} for {scores.numel()} scores'
        )
    selected = torch.argsort(scores, descending=True, stable=True)[:keep]
    return torch.sort(selected).values.to(dtype=torch.long)


def initialize_structured_width_pruned_state(
    model,
    source_state_dict,
    activation_second_moment,
):
    """Strictly transplant a factorized TLAST checkpoint to a smaller width.

    A single global hidden-channel subset is used throughout the network so
    residual additions and attention projections stay algebraically aligned.
    Predictor activation energy (estimated on the training split only) ranks
    the channels. The MAPA feed-forward hidden neurons are ranked separately
    by the product of their incoming and outgoing weight norms.
    """
    target_state = model.state_dict()
    required = {
        'data_encoding.linear1.weight',
        'spatial_agg_list.0.MLP2.0.weight',
        'spatial_agg_list.0.MLP2.2.weight',
        'time_wise_predictor.predict_unit.0.weight',
    }
    missing = sorted(required.difference(source_state_dict))
    if missing:
        raise KeyError(f'Structured-prune source is missing keys: {missing}')

    source_hid = int(
        source_state_dict['data_encoding.linear1.weight'].shape[0]
    )
    target_hid = int(target_state['data_encoding.linear1.weight'].shape[0])
    if target_hid >= source_hid:
        raise ValueError(
            'Structured width pruning requires target hid_dim < source '
            f'hid_dim, got {target_hid} >= {source_hid}'
        )

    second_moment = torch.as_tensor(
        activation_second_moment, dtype=torch.float64
    )
    predictor_input = source_state_dict[
        'time_wise_predictor.predict_unit.0.weight'
    ].shape[1]
    if second_moment.shape != (predictor_input, predictor_input):
        raise ValueError(
            'Activation second moment shape does not match the source '
            f'predictor input: {tuple(second_moment.shape)} != '
            f'({predictor_input}, {predictor_input})'
        )
    if predictor_input % source_hid != 0:
        raise ValueError(
            f'Predictor input {predictor_input} is not divisible by source '
            f'hid_dim {source_hid}'
        )
    input_steps = predictor_input // source_hid
    channel_scores = torch.diagonal(second_moment).reshape(
        input_steps, source_hid
    ).sum(dim=0)
    channel_index = _topk_sorted_indices(
        channel_scores, target_hid, 'hidden-channel'
    )
    history_index = torch.cat([
        channel_index + step * source_hid for step in range(input_steps)
    ])

    source_ff_in = source_state_dict[
        'spatial_agg_list.0.MLP2.0.weight'
    ]
    source_ff_out = source_state_dict[
        'spatial_agg_list.0.MLP2.2.weight'
    ]
    target_ff = target_state['spatial_agg_list.0.MLP2.0.weight'].shape[0]
    if source_ff_in.shape[0] != source_ff_out.shape[1]:
        raise ValueError('MAPA feed-forward source dimensions are inconsistent')
    ff_scores = (
        source_ff_in.double().norm(dim=1)
        * source_ff_out.double().norm(dim=0)
    )
    ff_index = _topk_sorted_indices(ff_scores, target_ff, 'MAPA-FF')

    branch_count = (
        source_state_dict['time_conv.fuse_conv.weight'].shape[1]
        // source_hid
    )
    fuse_index = torch.cat([
        channel_index + branch * source_hid
        for branch in range(branch_count)
    ])

    pruned = {}
    for key, target in target_state.items():
        if key not in source_state_dict:
            raise KeyError(f'Structured-prune source is missing {key!r}')
        source = source_state_dict[key]

        if source.shape == target.shape:
            value = source
        elif key == 'data_encoding.linear1.weight':
            value = source.index_select(0, channel_index)
        elif key == 'data_encoding.linear1.bias':
            value = source.index_select(0, channel_index)
        elif key in (
            'getTemb.time_in_day_embedding.weight',
            'getTemb.day_in_week_embedding.weight',
            'getSemb.spatial_embedding.weight',
            'spatial_agg_list.0.memory_bank',
        ):
            value = source.index_select(1, channel_index)
        elif key.startswith('time_conv.branches.') and key.endswith('.weight'):
            value = source.index_select(0, channel_index).index_select(
                1, channel_index
            )
        elif key.startswith('time_conv.branches.') and key.endswith('.bias'):
            value = source.index_select(0, channel_index)
        elif key == 'time_conv.fuse_conv.weight':
            value = source.index_select(0, channel_index).index_select(
                1, fuse_index
            )
        elif key == 'time_conv.fuse_conv.bias':
            value = source.index_select(0, channel_index)
        elif key == 'spatial_agg_list.0.MLP2.0.weight':
            value = source.index_select(0, ff_index).index_select(
                1, channel_index
            )
        elif key == 'spatial_agg_list.0.MLP2.0.bias':
            value = source.index_select(0, ff_index)
        elif key == 'spatial_agg_list.0.MLP2.2.weight':
            value = source.index_select(0, channel_index).index_select(
                1, ff_index
            )
        elif key == 'spatial_agg_list.0.MLP2.2.bias':
            value = source.index_select(0, channel_index)
        elif key == 'time_wise_predictor.predict_unit.0.weight':
            value = source.index_select(1, history_index)
        elif source.ndim == 2 and source.shape == (source_hid, source_hid):
            value = source.index_select(0, channel_index).index_select(
                1, channel_index
            )
        elif source.ndim == 1 and source.shape[0] == source_hid:
            value = source.index_select(0, channel_index)
        else:
            raise ValueError(
                f'No structured-prune rule for {key}: source '
                f'{tuple(source.shape)}, target {tuple(target.shape)}'
            )

        if value.shape != target.shape:
            raise ValueError(
                f'Structured-prune shape mismatch for {key}: produced '
                f'{tuple(value.shape)}, expected {tuple(target.shape)}'
            )
        pruned[key] = value.to(dtype=target.dtype, device=target.device)

    model.load_state_dict(pruned, strict=True)
    retained_activation = float(
        channel_scores[channel_index].sum()
        / channel_scores.sum().clamp_min(torch.finfo(torch.float64).eps)
    )
    return channel_index.tolist(), ff_index.tolist(), retained_activation

def main():
    # ===== 随机种子（Seed Ensemble / 可复现训练）=====
    # 固定所有随机源，使同一 --seed 可复现；3 个不同 seed 即 3 个不同初始化，构成集成。
    _seed = getattr(args, 'seed', 0)
    if _seed and _seed > 0:
        import random as _random
        _random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(_seed)

    if args.load_expid is not None and args.run_id is not None:
        raise ValueError('--run_id and --load_expid are mutually exclusive')
    if args.load_expid is None:
        if args.run_id is not None:
            expid = args.run_id
        else:
            expid = time.strftime("%m%d%H%M", time.localtime())
            if _seed and _seed > 0:
                expid = f"{expid}_s{_seed}"
    else:
        expid = args.load_expid

    inverse = True
    args.cache_data_path = 'datasets'
    
    if os.path.exists(f'./configurations/{args.config_file}'):
        with open(f'./configurations/{args.config_file}', 'r') as f:
            x = json.load(f)
            data_config = x['data_config']
            model_config = x['model_config']
            training_config = x['training_config']

            args.dataset_name =data_config['dataset_name']
            args.scaler_type = data_config.get('scaler_type', args.scaler_type)
            args.slice_size_per_day = data_config['slice_size_per_day']
            args.in_dim = data_config['in_dim']
            args.pre_dim = data_config['pre_dim']
            args.input_length = data_config['input_length']
            args.predict_length = data_config['predict_length']
            args.train_ratio = data_config['train_ratio']
            args.test_ratio = data_config['test_ratio']
            args.nodewise_scaler = bool(data_config.get('nodewise_scaler', 0))
            args.index_scaler_path = data_config.get(
                'index_scaler_path', 'none'
            )
            args.eval_mask = training_config['eval_mask']
            args.loss_type = training_config['loss_type']
            args.student_risk_loss_weight = float(
                training_config.get(
                    'student_risk_loss_weight',
                    args.student_risk_loss_weight,
                )
            )
            args.diff_consistency_weight = float(
                training_config.get('diff_consistency_weight', 0.0)
            )
            args.multires_block_loss_weight = float(
                training_config.get('multires_block_loss_weight', 0.0)
            )
            args.volatility_mae_strength = float(
                training_config.get('volatility_mae_strength', 0.0)
            )
            args.volatility_mae_quantile = float(
                training_config.get('volatility_mae_quantile', 0.75)
            )
            args.node_tail_risk_strength = float(
                training_config.get(
                    'node_tail_risk_strength',
                    args.node_tail_risk_strength,
                )
            )
            args.node_tail_risk_fraction = float(
                training_config.get(
                    'node_tail_risk_fraction',
                    args.node_tail_risk_fraction,
                )
            )
            args.node_tail_risk_scale_path = training_config.get(
                'node_tail_risk_scale_path',
                args.node_tail_risk_scale_path,
            )
            args.temporal_env_risk_strength = float(
                training_config.get(
                    'temporal_env_risk_strength',
                    args.temporal_env_risk_strength,
                )
            )
            args.temporal_env_count = int(
                training_config.get(
                    'temporal_env_count', args.temporal_env_count
                )
            )
            args.temporal_env_tail_fraction = float(
                training_config.get(
                    'temporal_env_tail_fraction',
                    args.temporal_env_tail_fraction,
                )
            )
            args.future_change_balance_strength = float(
                training_config.get(
                    'future_change_balance_strength',
                    args.future_change_balance_strength,
                )
            )
            args.future_change_balance_stats_path = training_config.get(
                'future_change_balance_stats_path',
                args.future_change_balance_stats_path,
            )
            args.change_persistence_loss_weight = float(
                training_config.get(
                    'change_persistence_loss_weight',
                    args.change_persistence_loss_weight,
                )
            )
            args.change_persistence_stats_path = training_config.get(
                'change_persistence_stats_path',
                args.change_persistence_stats_path,
            )
            args.change_event_loss_weight = float(
                training_config.get(
                    'change_event_loss_weight',
                    args.change_event_loss_weight,
                )
            )
            args.change_event_emphasis = float(
                training_config.get(
                    'change_event_emphasis',
                    args.change_event_emphasis,
                )
            )
            args.change_event_stats_path = training_config.get(
                'change_event_stats_path',
                args.change_event_stats_path,
            )
            args.batch_size = training_config['batch_size']
            args.learning_rate = training_config['learning_rate']
            args.neighbor_learning_rate = training_config.get(
                'neighbor_learning_rate', None
            )
            args.reranker_learning_rate = training_config.get(
                'reranker_learning_rate', None
            )
            args.lowrank_learning_rate = training_config.get(
                'lowrank_learning_rate', None
            )
            args.analog_utility_loss_weight = float(
                training_config.get('analog_utility_loss_weight', 0.0)
            )
            args.analog_utility_temperature = float(
                training_config.get('analog_utility_temperature', 1.0)
            )
            args.weight_decay = training_config['weight_decay']
            args.epochs = training_config['epochs']
            args.lr_scheduler_type = training_config['lr_scheduler_type']
            args.huber_delta = training_config['huber_delta']
            args.device = training_config.get('device', args.device)
            args.print_every = training_config.get('print_every', args.print_every)
            args.early_stop = bool(training_config.get('early_stop', args.early_stop))
            args.early_stop_step = training_config.get(
                'early_stop_step', args.early_stop_step
            )
            args.early_stop_start_epoch = training_config.get(
                'early_stop_start_epoch', args.early_stop_start_epoch
            )
            args.validation_only = bool(training_config.get(
                'validation_only', args.validation_only
            ))
            args.lr_decay = training_config.get('lr_decay', args.lr_decay)

            args.modelid = model_config["modelid"]
            args.M = model_config['M']
            args.hid_dim = model_config['hid_dim']
            args.n_heads = model_config['n_heads']
            args.num_layers = model_config['num_layers']
            args.tau = model_config['tau']
            args.d_out = model_config['d_out']
            args.emb_dropout = model_config['emb_dropout']
            args.se_emb_dropout = model_config['se_emb_dropout']
            args.te_emb_dropout = model_config['te_emb_dropout']
            args.tle_emb_dropout = model_config['tle_emb_dropout']
            args.enc_dropout = model_config.get('enc_dropout', args.enc_dropout)
            args.att_dropout = model_config.get('att_dropout', args.att_dropout)

            # ===== 加载新参数（MAPA、FTLE、MS-TCN）=====
            args.use_ftle = model_config.get('use_ftle', 1)
            args.use_time_lag_embedding = model_config.get(
                'use_time_lag_embedding', 1
            )
            args.ftle_mode = model_config.get('ftle_mode', 'full')
            args.single_tle_mode = model_config.get('single_tle_mode', 'last')
            args.use_mapa = model_config.get(
                'use_mapa', 1 if args.modelid in ('TLAST_improved', 'Scope') else 0
            )
            args.lambda_ortho = model_config.get('lambda_ortho', 0.01)
            args.num_prototypes = model_config.get('num_prototypes', 16)
            args.predictor_type = model_config.get('predictor_type', 'linear')
            args.predictor_factor_rank = int(
                model_config.get('predictor_factor_rank', 256)
            )
            args.predictor_activation_covariance = model_config.get(
                'predictor_activation_covariance', 'none'
            )
            args.predictor_activation_ridge = float(
                model_config.get('predictor_activation_ridge', 1e-5)
            )
            args.use_elastic_rank_router = int(model_config.get(
                'use_elastic_rank_router', 0
            ))
            args.elastic_low_rank = int(model_config.get(
                'elastic_low_rank', 64
            ))
            args.elastic_router_hidden = int(model_config.get(
                'elastic_router_hidden', 8
            ))
            args.elastic_route_mode = str(model_config.get(
                'elastic_route_mode', 'batch_topk'
            ))
            args.elastic_rank_temperature = float(model_config.get(
                'elastic_rank_temperature', 1.0
            ))
            args.elastic_initial_high_probability = float(model_config.get(
                'elastic_initial_high_probability', 0.8
            ))
            args.elastic_target_high_fraction = float(model_config.get(
                'elastic_target_high_fraction', 0.5
            ))
            args.elastic_budget_loss_weight = float(model_config.get(
                'elastic_budget_loss_weight', 1.0
            ))
            args.elastic_low_consistency_weight = float(model_config.get(
                'elastic_low_consistency_weight', 0.2
            ))
            args.use_student_risk_head = int(model_config.get(
                'use_student_risk_head', args.use_student_risk_head
            ))
            args.student_risk_hidden = int(model_config.get(
                'student_risk_hidden', args.student_risk_hidden
            ))
            args.use_signed_increment_mixer = int(model_config.get(
                'use_signed_increment_mixer',
                args.use_signed_increment_mixer,
            ))
            args.signed_increment_mixer_radius = int(model_config.get(
                'signed_increment_mixer_radius',
                args.signed_increment_mixer_radius,
            ))
            args.signed_increment_mixer_max_scale = float(model_config.get(
                'signed_increment_mixer_max_scale',
                args.signed_increment_mixer_max_scale,
            ))
            args.use_signed_change_encoding = int(model_config.get(
                'use_signed_change_encoding',
                args.use_signed_change_encoding,
            ))
            args.use_short_window_dynamics_encoding = int(model_config.get(
                'use_short_window_dynamics_encoding',
                args.use_short_window_dynamics_encoding,
            ))
            args.short_window_dynamics_encoding_max_scale = float(model_config.get(
                'short_window_dynamics_encoding_max_scale',
                args.short_window_dynamics_encoding_max_scale,
            ))
            args.use_sparse_neighbor_innovation = int(model_config.get(
                'use_sparse_neighbor_innovation', 0,
            ))
            args.neighbor_graph_path = model_config.get(
                'neighbor_graph_path', 'none',
            )
            args.signed_change_encoding_max_scale = float(model_config.get(
                'signed_change_encoding_max_scale',
                args.signed_change_encoding_max_scale,
            ))
            args.spatial_type = model_config.get(
                'spatial_type',
                'mapa' if args.modelid in ('TLAST_improved', 'Scope') else 'original_spa',
            )
            args.original_spa_dynamic_proxy = bool(
                model_config.get('original_spa_dynamic_proxy', True)
            )
            args.tcn_type = model_config.get('tcn_type', 'ms_tcn')
            args.prob_factor = int(model_config.get('prob_factor', 5))
            # MS-TCN 扩张卷积模式：支持字符串 "1,2,4" 或数组 [1,2,4]，None 表示原 tau_list 模式
            _dil = model_config.get('ms_tcn_dilations', None)
            if _dil is not None:
                if isinstance(_dil, str):
                    args.ms_tcn_dilations = [int(x.strip()) for x in _dil.split(',')]
                else:
                    args.ms_tcn_dilations = [int(x) for x in _dil]
            else:
                args.ms_tcn_dilations = None
            # 时空块堆叠模式（实验A）：[MS-TCN -> spatial] × num_layers 交替堆叠
            args.stack_blocks = bool(model_config.get('stack_blocks', False))
            args.ms_tcn_node_mix = bool(model_config.get('ms_tcn_node_mix', False))
            if model_config.get('time_weights', None) is not None:
                args.time_weights = model_config['time_weights']
            args.horizon_grad_balance_beta = float(model_config.get(
                'horizon_grad_balance_beta', args.horizon_grad_balance_beta
            ))
            args.horizon_grad_balance_momentum = float(model_config.get(
                'horizon_grad_balance_momentum',
                args.horizon_grad_balance_momentum,
            ))
            args.horizon_grad_balance_interval = int(model_config.get(
                'horizon_grad_balance_interval',
                args.horizon_grad_balance_interval,
            ))
            args.horizon_grad_balance_warmup_steps = int(model_config.get(
                'horizon_grad_balance_warmup_steps',
                args.horizon_grad_balance_warmup_steps,
            ))
            args.horizon_grad_balance_min_weight = float(model_config.get(
                'horizon_grad_balance_min_weight',
                args.horizon_grad_balance_min_weight,
            ))
            args.horizon_grad_balance_max_weight = float(model_config.get(
                'horizon_grad_balance_max_weight',
                args.horizon_grad_balance_max_weight,
            ))
            # 方案二：解耦常态/残差流（减输入每日均值 + 加回未来每日均值）
            args.base_residual = bool(model_config.get('base_residual', False))
            args.base_residual_gate = bool(model_config.get('base_residual_gate', False))
            if model_config.get('base_mean_path', None) is not None:
                args.base_mean_path = model_config['base_mean_path']
            # MAPA 残差绕过（ablation：验证 MAPA 16→307 压缩是否是真瓶颈）
            # 第一版固定 alpha（如 0.1），如有效再改 learnable
            args.mapa_bypass_alpha = float(model_config.get('mapa_bypass_alpha', 0.0))

            # ===== 补充：以下键原本仅靠 argparse 默认值，未从 config 读取，导致
            # norm_flag 等配置被静默忽略（例如 norm_first 配置始终以 'none' 运行）。
            # 统一在此从 model_config 读取，保持 config 真正可控。默认值与 argparse 一致，
            # 对现存 config 行为无影响。 =====
            args.norm_flag = model_config.get('norm_flag', 'none')
            args.att_type = model_config.get('att_type', 'proxy')
            args.activation_data = model_config.get('activation_data', 'relu')
            args.activation_enc = model_config.get('activation_enc', 'gelu')
            args.activation_dec = model_config.get('activation_dec', 'gelu')
            args.query_cross_time = model_config.get('query_cross_time', 1)
            args.return_att = model_config.get('return_att', 0)
            args.use_revin = model_config.get('use_revin', 0)

            # ===== EMA / SWA 权重平均（与 seed 正交，可组合）=====
            args.use_ema = model_config.get('use_ema', 0)
            args.ema_decay = model_config.get('ema_decay', 0.999)
            args.use_swa = model_config.get('use_swa', 0)
            args.swa_start_epoch = model_config.get('swa_start_epoch', 50)

            # ===== Node Dropout（节点/传感器级 dropout，与现有 6 个 dropout 正交）=====
            args.use_node_dropout = model_config.get('use_node_dropout', 0)
            args.node_dropout_p = model_config.get('node_dropout_p', 0.1)

            # ===== 2B: 并行节点嵌入（补回 MAPA 原型压缩丢失的节点身份）=====
            args.use_parallel_node_emb = model_config.get('use_parallel_node_emb', 0)
            args.parallel_node_emb_dropout = model_config.get('parallel_node_emb_dropout', 0.1)

            # ===== Future-time-conditioned richer predictor =====
            args.use_future_time_decoder = model_config.get('use_future_time_decoder', 0)

            # ===== Observable-volatility-conditioned forecast-delta scaling =====
            args.use_volatility_delta_scale = model_config.get(
                'use_volatility_delta_scale', 0
            )
            args.volatility_delta_hidden = model_config.get(
                'volatility_delta_hidden', 8
            )
            args.volatility_delta_max = model_config.get(
                'volatility_delta_max', 0.1
            )

            # ===== Zero-initialized node-specific linear dynamics bypass =====
            args.use_node_linear_dynamics = model_config.get(
                'use_node_linear_dynamics', 0
            )
            args.node_linear_dynamics_scale = model_config.get(
                'node_linear_dynamics_scale', 0.25
            )
            args.use_regime_gated_residual = model_config.get(
                'use_regime_gated_residual', 0
            )
            args.regime_residual_hidden = model_config.get(
                'regime_residual_hidden', 32
            )
            args.regime_residual_horizon_dim = model_config.get(
                'regime_residual_horizon_dim', 8
            )
            args.regime_residual_scale = model_config.get(
                'regime_residual_scale', 0.25
            )
            args.use_multires_block_residual = model_config.get(
                'use_multires_block_residual', 0
            )
            args.multires_block_size = model_config.get(
                'multires_block_size', 3
            )
            args.multires_block_hidden = model_config.get(
                'multires_block_hidden', 32
            )
            args.multires_block_scale = model_config.get(
                'multires_block_scale', 0.25
            )
            args.use_seasonal_drift_residual = model_config.get(
                'use_seasonal_drift_residual', 0
            )
            args.seasonal_drift_max = model_config.get(
                'seasonal_drift_max', 1.0
            )
            args.use_analog_retrieval_residual = model_config.get(
                'use_analog_retrieval_residual', 0
            )
            args.analog_retrieval_max = model_config.get(
                'analog_retrieval_max', 1.0
            )
            args.analog_retrieval_k = model_config.get(
                'analog_retrieval_k', 4
            )
            args.analog_retrieval_cache = model_config.get(
                'analog_retrieval_cache', 'none'
            )
            args.use_analog_candidate_reranker = model_config.get(
                'use_analog_candidate_reranker', 0
            )
            args.use_analog_direct_prediction = model_config.get(
                'use_analog_direct_prediction', 0
            )
            args.analog_candidate_feature_dim = model_config.get(
                'analog_candidate_feature_dim', 8
            )
            args.analog_candidate_disagreement_dim = model_config.get(
                'analog_candidate_disagreement_dim', 6
            )
            args.analog_candidate_hidden = model_config.get(
                'analog_candidate_hidden', 16
            )
            args.analog_candidate_parent_residual_features = model_config.get(
                'analog_candidate_parent_residual_features', 0
            )
            args.analog_candidate_parent_prediction_cache = model_config.get(
                'analog_candidate_parent_prediction_cache', 'none'
            )
            args.analog_candidate_level_align = model_config.get(
                'analog_candidate_level_align', 0
            )
            args.analog_candidate_distance_metric = model_config.get(
                'analog_candidate_distance_metric', 'masked_mse'
            )
            args.analog_candidate_selection_strategy = model_config.get(
                'analog_candidate_selection_strategy', 'nearest'
            )
            args.use_analog_code_domain_aggregation = model_config.get(
                'use_analog_code_domain_aggregation', 0
            )
            args.analog_candidate_code_cache = model_config.get(
                'analog_candidate_code_cache', 'none'
            )
            args.analog_code_domain_decode_candidates = model_config.get(
                'analog_code_domain_decode_candidates', 1
            )
            args.use_analog_correction_utility_gate = model_config.get(
                'use_analog_correction_utility_gate', 0
            )
            args.analog_correction_utility_hidden = model_config.get(
                'analog_correction_utility_hidden', 8
            )
            args.analog_correction_utility_node_rank = model_config.get(
                'analog_correction_utility_node_rank', 2
            )
            args.analog_correction_utility_max = model_config.get(
                'analog_correction_utility_max', 1.0
            )
            args.use_analog_adaptive_residual_mixer = model_config.get(
                'use_analog_adaptive_residual_mixer', 0
            )
            args.analog_adaptive_mixer_hidden = model_config.get(
                'analog_adaptive_mixer_hidden', 16
            )
            args.analog_adaptive_mixer_node_rank = model_config.get(
                'analog_adaptive_mixer_node_rank', 2
            )
            args.analog_adaptive_mixer_max = model_config.get(
                'analog_adaptive_mixer_max', 0.25
            )
            args.analog_candidate_min_valid_history = model_config.get(
                'analog_candidate_min_valid_history', 6
            )
            args.analog_candidate_cache = model_config.get(
                'analog_candidate_cache', 'none'
            )
            args.use_analog_query_conditioned_residual = model_config.get(
                'use_analog_query_conditioned_residual', 0
            )
            args.analog_query_conditioned_hidden = model_config.get(
                'analog_query_conditioned_hidden', 8
            )
            args.analog_query_conditioned_max = model_config.get(
                'analog_query_conditioned_max', 0.25
            )
            args.use_analog_lowrank_retrieval_residual = model_config.get(
                'use_analog_lowrank_retrieval_residual', 0
            )
            args.use_analog_temporal_lowrank_decoder = model_config.get(
                'use_analog_temporal_lowrank_decoder', 0
            )
            args.analog_temporal_decoder_rank = model_config.get(
                'analog_temporal_decoder_rank', 4
            )
            args.analog_temporal_decoder_max_scale = model_config.get(
                'analog_temporal_decoder_max_scale', 0.25
            )
            args.use_analog_temporal_gap_residual = model_config.get(
                'use_analog_temporal_gap_residual', 0
            )
            args.analog_temporal_gap_rank = model_config.get(
                'analog_temporal_gap_rank', 4
            )
            args.analog_temporal_gap_max = model_config.get(
                'analog_temporal_gap_max', 0.25
            )
            args.use_analog_node_horizon_retrieval_residual = (
                model_config.get(
                    'use_analog_node_horizon_retrieval_residual', 0
                )
            )
            args.analog_retrieval_aggregation = model_config.get(
                'analog_retrieval_aggregation', 'mean'
            )
            args.use_analog_confidence_aware_retrieval_residual = (
                model_config.get(
                    'use_analog_confidence_aware_retrieval_residual', 0
                )
            )
            args.analog_confidence_max = model_config.get(
                'analog_confidence_max', 0.25
            )
            args.analog_confidence_node_rank = model_config.get(
                'analog_confidence_node_rank', 0
            )
            args.analog_confidence_feature_dim = model_config.get(
                'analog_confidence_feature_dim', 3
            )
            args.use_analog_weekly_retrieval_residual = (
                model_config.get(
                    'use_analog_weekly_retrieval_residual', 0
                )
            )
            args.analog_weekly_retrieval_max = model_config.get(
                'analog_weekly_retrieval_max', 0.5
            )
            args.analog_weekly_rank = model_config.get(
                'analog_weekly_rank', 2
            )
            args.analog_weekly_retrieval_k = model_config.get(
                'analog_weekly_retrieval_k', 3
            )
            args.analog_weekly_retrieval_cache = model_config.get(
                'analog_weekly_retrieval_cache', 'none'
            )
            args.use_analog_metric_retrieval_residual = (
                model_config.get(
                    'use_analog_metric_retrieval_residual', 0
                )
            )
            args.analog_metric_retrieval_max = model_config.get(
                'analog_metric_retrieval_max', 0.5
            )
            args.analog_metric_rank = model_config.get(
                'analog_metric_rank', 2
            )
            args.analog_metric_retrieval_k = model_config.get(
                'analog_metric_retrieval_k', 4
            )
            args.analog_metric_retrieval_cache = model_config.get(
                'analog_metric_retrieval_cache', 'none'
            )
            args.analog_lowrank_rank = model_config.get(
                'analog_lowrank_rank', 2
            )
            args.use_analog_exact_match_residual = model_config.get(
                'use_analog_exact_match_residual', 0
            )
            args.analog_exact_match_max = model_config.get(
                'analog_exact_match_max', 1.0
            )
            args.analog_exact_min_valid_history = model_config.get(
                'analog_exact_min_valid_history', 6
            )
            args.freeze_parent_train_node_linear = model_config.get(
                'freeze_parent_train_node_linear', 0
            )
            args.freeze_parent_train_regime_residual = model_config.get(
                'freeze_parent_train_regime_residual', 0
            )
            args.freeze_parent_train_multires_block_residual = (
                model_config.get(
                    'freeze_parent_train_multires_block_residual', 0
                )
            )
            args.freeze_parent_train_seasonal_drift = model_config.get(
                'freeze_parent_train_seasonal_drift', 0
            )
            args.freeze_parent_train_analog_retrieval = model_config.get(
                'freeze_parent_train_analog_retrieval', 0
            )
            args.freeze_parent_train_analog_candidate_reranker = (
                model_config.get(
                    'freeze_parent_train_analog_candidate_reranker', 0
                )
            )
            args.freeze_parent_train_analog_query_conditioned_residual = (
                model_config.get(
                    'freeze_parent_train_analog_query_conditioned_residual', 0
                )
            )
            args.joint_train_lowrank_with_reranker = model_config.get(
                'joint_train_lowrank_with_reranker', 0
            )
            args.freeze_parent_train_analog_lowrank_retrieval = (
                model_config.get(
                    'freeze_parent_train_analog_lowrank_retrieval', 0
                )
            )
            args.freeze_parent_train_analog_correction_utility_gate = (
                model_config.get(
                    'freeze_parent_train_analog_correction_utility_gate', 0
                )
            )
            args.freeze_parent_train_analog_adaptive_residual_mixer = (
                model_config.get(
                    'freeze_parent_train_analog_adaptive_residual_mixer', 0
                )
            )
            args.freeze_parent_train_analog_temporal_decoder = (
                model_config.get(
                    'freeze_parent_train_analog_temporal_decoder', 0
                )
            )
            args.freeze_parent_train_analog_temporal_gap_residual = (
                model_config.get(
                    'freeze_parent_train_analog_temporal_gap_residual', 0
                )
            )
            args.freeze_parent_train_analog_node_horizon_retrieval = (
                model_config.get(
                    'freeze_parent_train_analog_node_horizon_retrieval',
                    0,
                )
            )
            args.freeze_parent_train_analog_confidence_aware_retrieval = (
                model_config.get(
                    'freeze_parent_train_analog_confidence_aware_retrieval', 0
                )
            )
            args.freeze_parent_train_analog_weekly_retrieval = (
                model_config.get(
                    'freeze_parent_train_analog_weekly_retrieval', 0
                )
            )
            args.freeze_parent_train_analog_metric_retrieval = (
                model_config.get(
                    'freeze_parent_train_analog_metric_retrieval', 0
                )
            )
            args.freeze_parent_train_analog_exact_match = (
                model_config.get(
                    'freeze_parent_train_analog_exact_match', 0
                )
            )
            args.freeze_parent_train_analog_joint_retrieval = (
                model_config.get(
                    'freeze_parent_train_analog_joint_retrieval', 0
                )
            )
            args.frozen_parent_ckpt = model_config.get(
                'frozen_parent_ckpt', 'none'
            )
            args.student_init_ckpt = model_config.get(
                'student_init_ckpt', args.frozen_parent_ckpt
            )
            args.freeze_analog_lowrank_blend = model_config.get(
                'freeze_analog_lowrank_blend', 0
            )
            args.freeze_analog_lowrank_blend_ckpt = model_config.get(
                'freeze_analog_lowrank_blend_ckpt', 'none'
            )
            args.lowrank_student_finetune = model_config.get(
                'lowrank_student_finetune',
                model_config.get('retrieval_aware_head_finetune', 0),
            )
            args.retrieval_aware_head_finetune = (
                args.lowrank_student_finetune
            )
            args.student_train_scope = model_config.get(
                'student_train_scope',
                'predictor'
                if model_config.get('retrieval_aware_full_predictor', 0)
                else 'readouts',
            )
            args.student_include_retrieval_adapters = model_config.get(
                'student_include_retrieval_adapters',
                0 if model_config.get('parent_lite_head_only', 0) else 1,
            )
            args.structured_width_prune_finetune = model_config.get(
                'structured_width_prune_finetune', 0
            )
            args.parent_lite_head_only = model_config.get(
                'parent_lite_head_only', 0
            )
            args.retrieval_aware_full_predictor = model_config.get(
                'retrieval_aware_full_predictor', 0
            )
            args.teacher_predictions_path = model_config.get(
                'teacher_predictions_path', 'none'
            )
            args.teacher_predictions_space = model_config.get(
                'teacher_predictions_space', 'normalized'
            )
            args.teacher_name = model_config.get(
                'teacher_name', 'unspecified'
            )
            args.teacher_predictions_strict_alignment = model_config.get(
                'teacher_predictions_strict_alignment', 0
            )
            args.teacher_predictions_required_splits = model_config.get(
                'teacher_predictions_required_splits', 'train,val,test'
            )
            args.kd_alpha = float(
                model_config.get('kd_alpha', 0.0)
            )

            # 解析 tau_list（支持字符串 "3,5,7" 或数组 [3,5,7]）
            tau_list = model_config.get('tau_list', None)
            if tau_list is not None:
                if isinstance(tau_list, str):
                    # 字符串格式："3,5,7"
                    args.tau_list = [int(x.strip()) for x in tau_list.split(',')]
                elif isinstance(tau_list, list):
                    # 数组格式：[3, 5, 7]
                    args.tau_list = [int(x) for x in tau_list]
                else:
                    args.tau_list = None
            else:
                args.tau_list = None

            # Graph convolution parameters
            args.use_graph_conv = model_config.get('use_graph_conv', 0)
            args.graph_conv_type = model_config.get('graph_conv_type', 'adaptive')
            args.pearson_adj_path = model_config.get(
                'pearson_adj_path', args.pearson_adj_path
            )
            k_hops_value = model_config.get('k_hops', '1,2')
            args.use_multi_scale_tcn = model_config.get("use_multi_scale_tcn", 0)
            args.use_improved_tle = model_config.get("use_improved_tle", 1)
            args.use_improved_proxy = model_config.get("use_improved_proxy", 1)
            if isinstance(k_hops_value, str):
                args.k_hops = [int(x.strip()) for x in k_hops_value.split(',')]
            else:
                args.k_hops = [int(x) for x in k_hops_value]
    else:
        print('there is no config file!')
        raise NotImplementedError('no config file!')

    device = torch.device(args.device)

    save_path = f'./EXP_experiments/{args.modelid}/{args.dataset_name}/{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp' + str(
        expid) + "/"
    log_path = save_path

    if args.run_id is not None and os.path.exists(save_path):
        existing_entries = os.listdir(save_path)
        if existing_entries:
            raise FileExistsError(
                'Refusing to overwrite an explicit --run_id directory: '
                f'{save_path}. Choose a new run id or remove/archive the '
                'existing run deliberately.'
            )

    if args.load_expid is None:
        log_filename = f'{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp{expid}.log'
    else:
        current = time.strftime("%m%d%H%M", time.localtime())
        log_filename = f'{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp{expid}_load_ON{current}.log'
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = get_logger(log_path, log_filename)
    exp_name = f'{args.modelid}_{args.dataset_name}'
    logger.info(exp_name)

    # ===== 方案 A+B: 显式标注实际生效的空间聚合层 =====
    # 由于方案 A 将 use_mapa 解耦为 MAPA 的唯一开关，
    # att_type 仅在 use_mapa=0 时生效（full/prob/difformer），
    # 因此在此统一记录 effective_spatial_agg，避免日志歧义。
    if args.modelid == 'TLAST':
        if args.att_type == 'proxy':
            args.effective_spatial_agg = "OriginalSPA(dynamic_proxy=True)"
        elif args.att_type == 'difformer':
            args.effective_spatial_agg = "DIFFormer"
        else:
            args.effective_spatial_agg = "Full_Prob"
    elif getattr(args, 'spatial_type', 'mapa') == 'original_spa':
        dynamic = bool(getattr(args, 'original_spa_dynamic_proxy', True))
        args.effective_spatial_agg = f"OriginalSPA(dynamic_proxy={dynamic})"
    elif getattr(args, 'spatial_type', 'mapa') == 'agcrn':
        args.effective_spatial_agg = "AGCRN"
    elif getattr(args, 'use_mapa', 0) == 1:
        args.effective_spatial_agg = "MAPA"
    else:
        _att = getattr(args, 'att_type', 'full')
        if _att == 'difformer':
            args.effective_spatial_agg = "DIFFormer"
        elif _att in ('full', 'prob'):
            args.effective_spatial_agg = "Full_Prob"
        else:
            args.effective_spatial_agg = "Full_Prob"  # 默认兜底

    # Persist evaluation/optimizer semantics so corrected runs are distinguishable
    # from historical runs that selected checkpoints via an unweighted batch mean.
    args.validation_aggregation = (
        'mean_horizon_valid_elements_all_samples_v2'
    )
    args.optimizer_type = 'AdamW'

    if args.load_expid is None:
        with open(f'{save_path}/config.pkl','wb') as f:
            pickle.dump(args,f)
        with open(f'{save_path}/config.json','w') as f:
            json.dump(vars(args), f)
    
    
    dataloader, num_nodes, scaler, num_test_samples = datasets_utils.load_dataset(args, logger)
    args.num_nodes = num_nodes
    args.prob_factor = getattr(args, 'prob_factor', 5)

    # Parse k_hops from string to list if needed
    if isinstance(args.k_hops, str):
        args.k_hops = [int(x) for x in args.k_hops.split(',')]

    # Load graph data only for branches that actually consume it. Historical
    # baseline runs loaded/logged a 307x307 matrix even though MAPA ignored it.
    needs_graph = bool(args.use_graph_conv) or args.spatial_type == 'agcrn'
    if needs_graph and args.pearson_adj_path is not None:
        pearson_adj = np.load(args.pearson_adj_path)
        args.pearson_adj = torch.FloatTensor(pearson_adj)
        logger.info(f"Loaded Pearson adjacency matrix from {args.pearson_adj_path}, shape: {pearson_adj.shape}")
    elif needs_graph and os.path.exists(f'datasets/{args.dataset_name}/{args.dataset_name}_pearson_adj.npy'):
        pearson_adj_path = f'datasets/{args.dataset_name}/{args.dataset_name}_pearson_adj.npy'
        pearson_adj = np.load(pearson_adj_path)
        args.pearson_adj = torch.FloatTensor(pearson_adj)
        logger.info(f"Loaded Pearson adjacency matrix from {pearson_adj_path}, shape: {pearson_adj.shape}")
    else:
        args.pearson_adj = None

    torch.cuda.empty_cache()

    logger.info(args)
    logger.info(args.note)

    engine = trainer(scaler= scaler, device=device, args=args, inverse=inverse)
    total_para_num, trainable_para_num = get_parameter_number(engine.model)

    # ===== 方案 A: 加载 STD-MAE 预训练编码器权重（仅 encoder/base 部分）=====
    # 仅当命令行显式传入 --load_stdmae_ckpt 且文件存在时生效，默认 'none' 不影响原流程。
    if getattr(args, 'load_stdmae_ckpt', 'none') not in (None, 'none', ''):
        _ckpt_path = args.load_stdmae_ckpt
        if os.path.exists(_ckpt_path):
            try:
                _sd = torch.load(_ckpt_path, map_location='cpu')
                # 兼容两种存法：① 直接是 base 状态字典；② 含 'base.' 前缀的预训练模型状态字典
                if any(k.startswith('base.') for k in _sd.keys()):
                    _base_sd = {k[len('base.'):]: v for k, v in _sd.items() if k.startswith('base.')}
                else:
                    _base_sd = _sd
                _missing, _unexpected = engine.model.load_state_dict(_base_sd, strict=False)
                logger.info(f"[STD-MAE] Loaded pretrained encoder from {_ckpt_path} "
                            f"(missing={len(_missing)}, unexpected={len(_unexpected)})")
                if _missing:
                    logger.info(f"[STD-MAE] missing keys: {_missing}")
                if _unexpected:
                    logger.info(f"[STD-MAE] unexpected keys: {_unexpected}")
            except Exception as e:
                logger.error(f"[STD-MAE] Failed to load pretrained encoder from {_ckpt_path}: {e}")
        else:
            logger.warning(f"[STD-MAE] ckpt not found: {_ckpt_path}, skip loading")

    # ===== Frozen-parent residual-adapter fitting =====
    # This remains a single raw model checkpoint: the parent and trained
    # branch are saved together, and Test calls the ordinary model forward
    # without validation calibration or prediction ensembling.
    _frozen_adapter_names = engine.frozen_adapter_names
    _retrieval_aware_head_finetune = bool(
        getattr(args, 'lowrank_student_finetune', 0)
        or getattr(args, 'retrieval_aware_head_finetune', 0)
    )
    _structured_width_prune_finetune = bool(
        getattr(args, 'structured_width_prune_finetune', 0)
    )
    if _structured_width_prune_finetune and _retrieval_aware_head_finetune:
        raise ValueError(
            'structured_width_prune_finetune and '
            'retrieval_aware_head_finetune are mutually exclusive'
        )
    if _structured_width_prune_finetune and _frozen_adapter_names:
        raise ValueError(
            'structured_width_prune_finetune trains the complete student and '
            'cannot be combined with a frozen-adapter mode'
        )
    if _frozen_adapter_names and _retrieval_aware_head_finetune:
        raise ValueError(
            'retrieval_aware_head_finetune is a complete final-stage mode and '
            'cannot be combined with a frozen-adapter mode'
        )
    if _frozen_adapter_names:
        for _frozen_adapter_name in _frozen_adapter_names:
            if not hasattr(engine.model, _frozen_adapter_name):
                raise ValueError(
                    f'Frozen adapter {_frozen_adapter_name!r} is not enabled '
                    'in the model configuration'
                )
        _parent_ckpt = getattr(args, 'frozen_parent_ckpt', 'none')
        if _parent_ckpt in (None, '', 'none') or not os.path.exists(_parent_ckpt):
            raise FileNotFoundError(
                f'Frozen parent checkpoint not found: {_parent_ckpt}'
            )
        _parent_sd = torch.load(_parent_ckpt, map_location=device)
        # Backward-compatible expansion of the confidence gate from the
        # legacy three quality features to the six-feature gap-aware variant.
        # Preserve all learned legacy coefficients and zero-initialize only
        # the newly added feature columns, so epoch 0 exactly reproduces the
        # parent confidence function.
        _model_sd = engine.model.state_dict()
        _confidence_expand_keys = (
            'analog_confidence_aware_retrieval_residual.quality_weight',
            'analog_confidence_aware_retrieval_residual.node_quality_factor',
        )
        for _key in _confidence_expand_keys:
            if _key not in _parent_sd or _key not in _model_sd:
                continue
            _source = _parent_sd[_key]
            _target = _model_sd[_key]
            if _source.shape == _target.shape:
                continue
            _expandable = (
                _source.ndim == _target.ndim
                and _source.shape[0] == _target.shape[0]
                and _source.shape[1] == 3
                and _target.shape[1] == 6
                and _source.shape[2:] == _target.shape[2:]
            )
            if not _expandable:
                raise RuntimeError(
                    f'Unsupported confidence checkpoint migration for {_key}: '
                    f'{tuple(_source.shape)} -> {tuple(_target.shape)}'
                )
            _expanded = torch.zeros_like(_target)
            _expanded[:, :3, ...].copy_(_source)
            _parent_sd[_key] = _expanded
            logger.info(
                '[FrozenParent] expanded %s from %s to %s; new features zeroed',
                _key, tuple(_source.shape), tuple(_target.shape),
            )
        _reranker_key = (
            'analog_candidate_reranker.disagreement_projection.weight'
        )
        if _reranker_key in _parent_sd and _reranker_key in _model_sd:
            _source = _parent_sd[_reranker_key]
            _target = _model_sd[_reranker_key]
            if _source.shape != _target.shape:
                _expandable = (
                    _source.ndim == 2
                    and _target.ndim == 2
                    and _source.shape[0] == _target.shape[0]
                    and _source.shape[1] == 2
                    and _target.shape[1] == 6
                )
                if not _expandable:
                    raise RuntimeError(
                        'Unsupported reranker checkpoint migration for '
                        f'{_reranker_key}: {tuple(_source.shape)} -> '
                        f'{tuple(_target.shape)}'
                    )
                _expanded = torch.zeros_like(_target)
                _expanded[:, :2].copy_(_source)
                _parent_sd[_reranker_key] = _expanded
                logger.info(
                    '[FrozenParent] expanded %s from %s to %s; '
                    'new features zeroed',
                    _reranker_key, tuple(_source.shape), tuple(_target.shape),
                )
        _missing, _unexpected = engine.model.load_state_dict(
            _parent_sd, strict=False
        )
        _expected_missing = set()
        for _frozen_adapter_name in _frozen_adapter_names:
            _adapter_prefix = _frozen_adapter_name + '.'
            _module = getattr(
                engine.model, _frozen_adapter_name
            )
            for _key in _module.state_dict().keys():
                _full_key = _adapter_prefix + _key
                # A frozen adapter key is expected to be missing only if the
                # parent checkpoint does not already contain it (i.e. the
                # module is brand new in this session and must be randomly
                # initialised).  When the parent checkpoint already holds the
                # key (e.g. lowrank from a previous training stage is being
                # re-opened), it is expected to be loaded rather than missing.
                if _full_key not in _parent_sd:
                    _expected_missing.add(_full_key)
        if (
            set(_missing) != _expected_missing
            or list(_unexpected)
        ):
            raise RuntimeError(
                'Frozen parent is not architecture-compatible: '
                f'expected_missing={sorted(_expected_missing)}, '
                f'missing={list(_missing)}, unexpected={list(_unexpected)}'
            )

        for _parameter in engine.model.parameters():
            _parameter.requires_grad_(False)
        for _frozen_adapter_name in _frozen_adapter_names:
            for _parameter in getattr(
                engine.model, _frozen_adapter_name
            ).parameters():
                _parameter.requires_grad_(True)

        # Optional: pin the low-rank retrieval blend g_hn at its parent value.
        # This isolates whether a residual-construction gain is cancelled by
        # the joint optimizer shrinking the blend during co-adaptation.  Only
        # the three parameters that form g are frozen; every other trainable
        # module keeps learning.
        _freeze_lowrank_blend = bool(
            getattr(args, 'freeze_analog_lowrank_blend', 0)
        )
        if _freeze_lowrank_blend:
            if not hasattr(
                engine.model, 'analog_lowrank_retrieval_residual'
            ):
                raise ValueError(
                    'freeze_analog_lowrank_blend requires '
                    'analog_lowrank_retrieval_residual in the model'
                )
            _lowrank = engine.model.analog_lowrank_retrieval_residual
            # Optionally pin g at the value a canonical reference run
            # converged to (e.g. joint3) rather than at the pre-joint parent
            # value, so freezing does not silently discard the blend increase
            # the canonical pipeline itself would have learned.
            _blend_ckpt = getattr(
                args, 'freeze_analog_lowrank_blend_ckpt', 'none'
            )
            if _blend_ckpt not in (None, '', 'none'):
                if not os.path.exists(_blend_ckpt):
                    raise FileNotFoundError(
                        'freeze_analog_lowrank_blend_ckpt not found: '
                        f'{_blend_ckpt}'
                    )
                _blend_sd = torch.load(_blend_ckpt, map_location=device)
                _blend_prefix = 'analog_lowrank_retrieval_residual.'
                with torch.no_grad():
                    for _blend_name in (
                        'horizon_logits', 'node_factor', 'horizon_factor'
                    ):
                        _key = _blend_prefix + _blend_name
                        if _key not in _blend_sd:
                            raise KeyError(
                                f'{_key} missing from {_blend_ckpt}'
                            )
                        _target = getattr(_lowrank, _blend_name)
                        if _target.shape != _blend_sd[_key].shape:
                            raise ValueError(
                                f'shape mismatch for {_key}: '
                                f'{tuple(_target.shape)} != '
                                f'{tuple(_blend_sd[_key].shape)}'
                            )
                        _target.copy_(_blend_sd[_key])
                logger.info(
                    '[FixedBlend] g_hn overridden from %s', _blend_ckpt
                )
            with torch.no_grad():
                _g_logits = _lowrank.horizon_logits[:, None] + torch.einsum(
                    'nr,hr->hn', _lowrank.node_factor, _lowrank.horizon_factor
                )
                _g = _lowrank.max_blend * torch.tanh(_g_logits)
                logger.info(
                    '[FixedBlend] pinned g_hn: mean=%.6f min=%.6f max=%.6f '
                    '| per-horizon mean=%s',
                    float(_g.mean()), float(_g.min()), float(_g.max()),
                    [round(float(v), 6) for v in _g.mean(dim=1)],
                )
            for _blend_name in (
                'horizon_logits', 'node_factor', 'horizon_factor'
            ):
                getattr(_lowrank, _blend_name).requires_grad_(False)
            logger.info(
                '[FixedBlend] frozen analog_lowrank_retrieval_residual.'
                '{horizon_logits,node_factor,horizon_factor}; '
                'g_hn is constant from here on'
            )

        _trainable_parameters = [
            parameter for parameter in engine.model.parameters()
            if parameter.requires_grad
        ]
        if set(_frozen_adapter_names) == {
            'analog_candidate_reranker',
            'analog_lowrank_retrieval_residual',
        }:
            _reranker_lr = (
                args.reranker_learning_rate
                if args.reranker_learning_rate is not None
                else args.learning_rate
            )
            _lowrank_lr = (
                args.lowrank_learning_rate
                if args.lowrank_learning_rate is not None
                else args.learning_rate
            )
            # Only parameters that survived the fixed-blend freeze may be
            # handed to the optimizer, otherwise AdamW would track tensors
            # whose grad is permanently None.
            _param_groups = []
            _reranker_params = [
                p for p in
                engine.model.analog_candidate_reranker.parameters()
                if p.requires_grad
            ]
            _lowrank_params = [
                p for p in
                engine.model.analog_lowrank_retrieval_residual.parameters()
                if p.requires_grad
            ]
            if _reranker_params:
                _param_groups.append(
                    {'params': _reranker_params, 'lr': _reranker_lr}
                )
            if _lowrank_params:
                _param_groups.append(
                    {'params': _lowrank_params, 'lr': _lowrank_lr}
                )
            if not _param_groups:
                raise RuntimeError(
                    'Frozen-parent fitting selected no trainable parameters'
                )
            engine.optimizer = optim.AdamW(
                _param_groups,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            logger.info(
                '[FrozenParent] grouped learning rates: reranker=%g, lowrank=%g',
                _reranker_lr, _lowrank_lr,
            )
        else:
            engine.optimizer = optim.AdamW(
                _trainable_parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        engine.lr_scheduler = engine._build_lr_scheduler()
        _total, _trainable = get_parameter_number(engine.model)
        logger.info(
            '[FrozenParent] adapters=%s, loaded=%s, missing=%s, unexpected=%s, '
            'total_parameters=%d, trainable_parameters=%d',
            list(_frozen_adapter_names), _parent_ckpt,
            list(_missing), list(_unexpected),
            _total, _trainable,
        )

    # ===== Activation-guided structured width pruning =====
    # This changes actual tensor dimensions (rather than installing a sparse
    # mask), so both parameter count and dense inference FLOPs fall. The
    # complete smaller model is then fine-tuned with optional KD.
    if _structured_width_prune_finetune:
        _parent_ckpt = getattr(args, 'frozen_parent_ckpt', 'none')
        _covariance_path = getattr(
            args, 'predictor_activation_covariance', 'none'
        )
        if _parent_ckpt in (None, '', 'none') or not os.path.exists(_parent_ckpt):
            raise FileNotFoundError(
                f'Structured-prune parent not found: {_parent_ckpt}'
            )
        if (
            _covariance_path in (None, '', 'none')
            or not os.path.exists(_covariance_path)
        ):
            raise FileNotFoundError(
                'Structured width pruning requires a train-only activation '
                f'second moment: {_covariance_path}'
            )
        _parent_sd = torch.load(_parent_ckpt, map_location='cpu')
        with np.load(_covariance_path, allow_pickle=False) as _archive:
            if 'second_moment' not in _archive:
                raise KeyError(
                    'Activation covariance archive must contain "second_moment"'
                )
            _second_moment = np.asarray(
                _archive['second_moment'], dtype=np.float64
            )
        _channels, _ff_neurons, _retained_activation = (
            initialize_structured_width_pruned_state(
                engine.model, _parent_sd, _second_moment
            )
        )
        for _parameter in engine.model.parameters():
            _parameter.requires_grad_(True)
        engine.optimizer = optim.AdamW(
            engine.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        engine.lr_scheduler = engine._build_lr_scheduler()
        _total, _trainable = get_parameter_number(engine.model)
        logger.info(
            '[StructuredWidthPrune] loaded=%s, hid_dim=%d, '
            'retained_activation=%.8f, channels=%s, MAPA_FF_neurons=%d, '
            'total_parameters=%d, trainable_parameters=%d',
            _parent_ckpt, args.hid_dim, _retained_activation, _channels,
            len(_ff_neurons), _total, _trainable,
        )

    # ===== Teacher-agnostic low-rank student fine-tuning =====
    # Initialization and distillation are deliberately decoupled: the former
    # needs an architecture-compatible dense checkpoint, while the latter
    # consumes predictions from any teacher through the cache contract.
    # Optional retrieval adapters are retained only for backward-compatible
    # retrieval-based experiments. Strict loading and an epoch-0 Validation anchor
    # prevent silent initialization drift or Test-driven selection.
    if _retrieval_aware_head_finetune:
        _parent_ckpt = getattr(args, 'student_init_ckpt', 'none')
        if _parent_ckpt in (None, '', 'none'):
            _parent_ckpt = getattr(args, 'frozen_parent_ckpt', 'none')
        if _parent_ckpt in (None, '', 'none') or not os.path.exists(_parent_ckpt):
            raise FileNotFoundError(
                f'Low-rank student initialization checkpoint not found: '
                f'{_parent_ckpt}'
            )
        _parent_lite_head_only = bool(getattr(
            args, 'parent_lite_head_only', 0
        ))
        _student_train_scope = str(getattr(
            args, 'student_train_scope',
            'predictor'
            if getattr(args, 'retrieval_aware_full_predictor', 0)
            else 'readouts',
        )).lower()
        if _student_train_scope not in ('readouts', 'predictor', 'all'):
            raise ValueError(
                'student_train_scope must be readouts, predictor, or all; '
                f'got {_student_train_scope!r}'
            )
        _include_retrieval_adapters = bool(getattr(
            args, 'student_include_retrieval_adapters',
            not _parent_lite_head_only,
        ))
        required_modules = ('time_wise_predictor',)
        if _include_retrieval_adapters:
            required_modules += (
                'analog_lowrank_retrieval_residual',
                'analog_exact_match_residual',
            )
        for _module_name in required_modules:
            if not hasattr(engine.model, _module_name):
                raise ValueError(
                    'lowrank_student_finetune requires '
                    f'{_module_name!r} in the active model configuration'
                )
        _predictor = engine.model.time_wise_predictor
        if not hasattr(_predictor, 'predict_unit_list'):
            raise ValueError(
                'lowrank_student_finetune requires the richer predictor '
                'with predict_unit_list final readouts'
            )
        _parent_sd = torch.load(_parent_ckpt, map_location=device)
        _factorized_parent_key = (
            'time_wise_predictor.predict_unit.1.weight'
        )
        _same_factorized_parent = (
            isinstance(_predictor, Scope.factorized_richer_predictor)
            and _factorized_parent_key in _parent_sd
            and tuple(_parent_sd[_factorized_parent_key].shape)
            == tuple(_predictor.predict_unit[1].weight.shape)
        )
        if _same_factorized_parent:
            _missing, _unexpected = engine.model.load_state_dict(
                _parent_sd, strict=False
            )
            _expected_missing = set()
            if bool(getattr(args, 'use_elastic_rank_router', 0)):
                _expected_missing.update({
                    'elastic_rank_router.' + key
                    for key in engine.model.elastic_rank_router.state_dict()
                })
                _expected_missing.add(
                    'time_wise_predictor.elastic_routing_threshold'
                )
            if set(_missing) != _expected_missing or list(_unexpected):
                raise RuntimeError(
                    'Factorized Student checkpoint mismatch: '
                    f'missing={list(_missing)}, unexpected={list(_unexpected)}'
                )
            logger.info(
                '[ElasticRank] loaded architecture-compatible rank-%d '
                'Student; new router parameters=%d',
                _predictor.factor_rank,
                sum(parameter.numel() for parameter in
                    engine.model.elastic_rank_router.parameters())
                if bool(getattr(args, 'use_elastic_rank_router', 0)) else 0,
            )
        elif isinstance(
            _predictor, Scope.factorized_richer_predictor
        ):
            # Load every non-predictor tensor strictly, then migrate the
            # legacy richer head through truncated SVD and exact step-bias
            # folding. This prevents strict=False from silently hiding a
            # missing backbone or retrieval tensor.
            _shared_parent_sd = {
                key: value for key, value in _parent_sd.items()
                if not key.startswith('time_wise_predictor.')
            }
            _missing, _unexpected = engine.model.load_state_dict(
                _shared_parent_sd, strict=False
            )
            _expected_predictor_missing = {
                'time_wise_predictor.' + key
                for key in _predictor.state_dict().keys()
            }
            if bool(getattr(args, 'use_student_risk_head', 0)):
                _expected_predictor_missing.update({
                    'student_risk_head.' + key
                    for key in engine.model.student_risk_head.state_dict().keys()
                })
            if bool(getattr(args, 'use_signed_change_encoding', 0)):
                _expected_predictor_missing.update({
                    'signed_change_encoding.' + key
                    for key in engine.model.signed_change_encoding.state_dict()
                })
            if bool(getattr(args, 'use_short_window_dynamics_encoding', 0)):
                _expected_predictor_missing.update({
                    'short_window_dynamics_encoding.' + key
                    for key in engine.model.short_window_dynamics_encoding.state_dict()
                })
            if bool(getattr(args, 'use_sparse_neighbor_innovation', 0)):
                _expected_predictor_missing.update({
                    'sparse_neighbor_innovation.' + key
                    for key in engine.model.sparse_neighbor_innovation.state_dict()
                })
            if (
                set(_missing) != _expected_predictor_missing
                or list(_unexpected)
            ):
                raise RuntimeError(
                    'Lite predictor parent is not architecture-compatible: '
                    f'expected predictor-only missing keys='
                    f'{sorted(_expected_predictor_missing)}, '
                    f'missing={list(_missing)}, '
                    f'unexpected={list(_unexpected)}'
                )
            _activation_second_moment = None
            _activation_covariance_path = getattr(
                args, 'predictor_activation_covariance', 'none'
            )
            if _activation_covariance_path not in (None, '', 'none'):
                if not os.path.exists(_activation_covariance_path):
                    raise FileNotFoundError(
                        'Predictor activation covariance not found: '
                        f'{_activation_covariance_path}'
                    )
                with np.load(
                    _activation_covariance_path, allow_pickle=False
                ) as _activation_archive:
                    if 'second_moment' not in _activation_archive:
                        raise KeyError(
                            'Activation covariance archive must contain '
                            '"second_moment"'
                        )
                    _activation_second_moment = np.asarray(
                        _activation_archive['second_moment'],
                        dtype=np.float64,
                    )
                logger.info(
                    '[LitePredictor] activation-aware initialization from %s',
                    _activation_covariance_path,
                )
            _retained_energy = _predictor.initialize_from_richer_state_dict(
                _parent_sd,
                prefix='time_wise_predictor.',
                activation_second_moment=_activation_second_moment,
                activation_ridge=float(getattr(
                    args, 'predictor_activation_ridge', 1e-5
                )),
            )
            logger.info(
                '[LitePredictor] SVD rank=%d, retained weight energy=%.8f; '
                'legacy step embedding folded exactly into horizon biases',
                _predictor.factor_rank, _retained_energy,
            )
        else:
            engine.model.load_state_dict(_parent_sd, strict=True)

        for _parameter in engine.model.parameters():
            _parameter.requires_grad_(False)
        if _student_train_scope == 'all':
            _partial_modules = [engine.model]
        elif _student_train_scope == 'predictor':
            _partial_modules = [_predictor]
        else:
            _partial_modules = [_predictor.predict_unit_list]
        if bool(getattr(args, 'use_elastic_rank_router', 0)):
            _partial_modules.append(engine.model.elastic_rank_router)
        if bool(getattr(args, 'use_signed_change_encoding', 0)):
            if not hasattr(engine.model, 'signed_change_encoding'):
                raise RuntimeError(
                    'use_signed_change_encoding=1 but the model has no '
                    'signed_change_encoding module'
                )
            _partial_modules.append(engine.model.signed_change_encoding)
        if bool(getattr(args, 'use_short_window_dynamics_encoding', 0)):
            _partial_modules.append(engine.model.short_window_dynamics_encoding)
        if bool(getattr(args, 'use_sparse_neighbor_innovation', 0)):
            _partial_modules.append(engine.model.sparse_neighbor_innovation)
        if bool(getattr(args, 'use_student_risk_head', 0)):
            if not hasattr(engine.model, 'student_risk_head'):
                raise RuntimeError(
                    'use_student_risk_head=1 but the model has no risk head'
                )
            _partial_modules.append(engine.model.student_risk_head)
        if _include_retrieval_adapters:
            _partial_modules.extend([
                engine.model.analog_lowrank_retrieval_residual,
                engine.model.analog_exact_match_residual,
            ])
        for _module in _partial_modules:
            for _parameter in _module.parameters():
                _parameter.requires_grad_(True)
        engine.partial_finetune_modules = _partial_modules
        _trainable_parameters = [
            parameter for parameter in engine.model.parameters()
            if parameter.requires_grad
        ]
        if not _trainable_parameters:
            raise RuntimeError(
                'lowrank_student_finetune selected no trainable parameters'
            )
        _neighbor_learning_rate = getattr(
            args, 'neighbor_learning_rate', None
        )
        if (
            _neighbor_learning_rate is not None
            and bool(getattr(args, 'use_sparse_neighbor_innovation', 0))
        ):
            _neighbor_parameters = list(
                engine.model.sparse_neighbor_innovation.parameters()
            )
            _neighbor_parameter_ids = {
                id(parameter) for parameter in _neighbor_parameters
            }
            _base_parameters = [
                parameter for parameter in _trainable_parameters
                if id(parameter) not in _neighbor_parameter_ids
            ]
            engine.optimizer = optim.AdamW(
                [
                    {
                        'params': _base_parameters,
                        'lr': args.learning_rate,
                    },
                    {
                        'params': _neighbor_parameters,
                        'lr': float(_neighbor_learning_rate),
                    },
                ],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            logger.info(
                '[SparseNeighborInnovation] optimizer groups: '
                'base_lr=%.8g (%d tensors), neighbor_lr=%.8g (%d tensors)',
                args.learning_rate, len(_base_parameters),
                float(_neighbor_learning_rate), len(_neighbor_parameters),
            )
        else:
            engine.optimizer = optim.AdamW(
                _trainable_parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        engine.lr_scheduler = engine._build_lr_scheduler()
        _total, _trainable = get_parameter_number(engine.model)
        logger.info(
            '[LowRankStudentFT] init=%s strict=True, scope=%s, '
            'teacher=%s, modules=%s, total_parameters=%d, '
            'trainable_parameters=%d',
            _parent_ckpt, _student_train_scope,
            getattr(args, 'teacher_name', 'unspecified'),
            [
                'model'
                if _student_train_scope == 'all'
                else (
                    'time_wise_predictor'
                    if _student_train_scope == 'predictor'
                    else 'time_wise_predictor.predict_unit_list'
                )
            ] + ([] if not _include_retrieval_adapters else [
                'analog_lowrank_retrieval_residual',
                'analog_exact_match_residual',
            ]) + ([] if not bool(getattr(
                args, 'use_signed_change_encoding', 0
            )) else ['signed_change_encoding']) + ([] if not bool(getattr(
                args, 'use_short_window_dynamics_encoding', 0
            )) else ['short_window_dynamics_encoding']),
            _total, _trainable,
        )

    # ===== EMA / SWA 权重平均初始化（与 seed 正交）=====
    # 二者互斥：use_swa 优先。EMA 全程跟踪；SWA 仅从 swa_start_epoch 起平均。
    ema_shadow = None
    swa_model = None
    swa_start_epoch = args.swa_start_epoch
    if args.use_swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(engine.model)
        logger.info(f"[SWA] enabled, start averaging from epoch {swa_start_epoch}")
    elif args.use_ema:
        ema_decay = args.ema_decay
        # 仅对可学习参数做 EMA；buffers 保持不变
        ema_shadow = {n: p.detach().clone() for n, p in engine.model.named_parameters()}
        logger.info(f"[EMA] enabled, decay={ema_decay}")

    # ===== 方案 A: 显式打印实际生效的空间聚合层，与 effective_spatial_agg 交叉验证 =====
    _actual_layer = type(engine.model.spatial_agg_list[0]).__name__
    logger.info(f"[Spatial Aggregator] effective={args.effective_spatial_agg} | actual_layer={_actual_layer}")
    if args.effective_spatial_agg == 'MAPA':
        assert _actual_layer == 'MAPA_EncoderLayer'
    elif args.effective_spatial_agg.startswith('OriginalSPA'):
        assert _actual_layer == 'EncoderLayer'
    elif args.effective_spatial_agg == 'AGCRN':
        assert _actual_layer == 'AGCRNSpatial'
    elif args.effective_spatial_agg == 'DIFFormer':
        assert _actual_layer == 'DIFFormer_EncoderLayer'
    else:
        assert _actual_layer == 'Full_Prob_EncoderLayer'
    
    if args.load_expid is None:
        logger.info("start training...")
        all_start_time = time.time()
        his_loss = []
        trainloss_record = []
        val_time = []
        train_time = []
        best_validate_loss = np.inf
        validate_score_non_decrease_count = 0

        log_in_train_details = []
        batches_seen = 0

        # In frozen-parent mode, epoch 0 is the exact parent forecast because
        # the added branch is zero-initialized.  Evaluate and save it before
        # any update so branch fitting can never silently replace the parent
        # with a worse validation checkpoint.
        if (
            _frozen_adapter_names
            or _retrieval_aware_head_finetune
            or _structured_width_prune_finetune
        ):
            valid_totals = {
                metric: np.zeros(args.predict_length, dtype=np.float64)
                for metric in ('loss', 'mae', 'mape', 'mse')
            }
            valid_counts = {
                metric: np.zeros(args.predict_length, dtype=np.float64)
                for metric in ('loss', 'mae', 'mape', 'rmse')
            }
            _epoch0_start = time.time()
            for x, y in dataloader['val_loader'].get_iterator():
                (
                    x, x_time, analog_retrieval,
                    exact_retrieval, exact_available,
                    candidate_future, candidate_available,
                    candidate_features,
                ) = unpack_dataloader_input(x)
                y, y_time = y
                valx = [
                    torch.Tensor(x).to(device),
                    torch.LongTensor(x_time).to(device),
                    torch.LongTensor(y_time).to(device),
                ]
                if analog_retrieval is not None:
                    valx.append(
                        torch.as_tensor(
                            analog_retrieval,
                            dtype=(
                                torch.float16
                                if analog_retrieval.dtype == np.float16
                                else torch.float32
                            ),
                            device=device,
                        )
                    )
                if exact_retrieval is not None:
                    cache_float_dtype = (
                        torch.float16
                        if exact_retrieval.dtype == np.float16
                        else torch.float32
                    )
                    valx.extend([
                        torch.as_tensor(
                            exact_retrieval,
                            dtype=cache_float_dtype,
                            device=device,
                        ),
                        torch.as_tensor(
                            exact_available,
                            dtype=cache_float_dtype,
                            device=device,
                        ),
                    ])
                if candidate_future is not None:
                    valx.extend([
                        torch.as_tensor(
                            candidate_future,
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            candidate_available,
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            candidate_features,
                            dtype=torch.float32,
                            device=device,
                        ),
                    ])
                valy = torch.Tensor(y).to(device)
                metrics = engine.eval(valx, valy, return_counts=True)
                horizon = metrics[4]['horizon']
                loss_values = np.asarray(horizon['loss'])
                mae_values = np.asarray(horizon['mae'])
                mape_values = np.asarray(horizon['mape'])
                rmse_values = np.asarray(horizon['rmse'])
                valid_totals['loss'] += (
                    loss_values * np.asarray(horizon['loss_count'])
                )
                valid_totals['mae'] += (
                    mae_values * np.asarray(horizon['mae_count'])
                )
                valid_totals['mape'] += (
                    mape_values * np.asarray(horizon['mape_count'])
                )
                valid_totals['mse'] += (
                    rmse_values ** 2
                    * np.asarray(horizon['rmse_count'])
                )
                for key in valid_counts:
                    valid_counts[key] += np.asarray(
                        horizon[f'{key}_count']
                    )

            horizon_loss = valid_totals['loss'] / np.maximum(
                valid_counts['loss'], 1.0
            )
            horizon_mae = valid_totals['mae'] / np.maximum(
                valid_counts['mae'], 1.0
            )
            horizon_mape = valid_totals['mape'] / np.maximum(
                valid_counts['mape'], 1.0
            )
            horizon_rmse = np.sqrt(
                valid_totals['mse']
                / np.maximum(valid_counts['rmse'], 1.0)
            )
            _epoch0_metrics = [
                float(horizon_loss.mean()),
                float(horizon_mae.mean()),
                float(horizon_mape.mean()),
                float(horizon_rmse.mean()),
            ]
            _epoch0_time = time.time() - _epoch0_start
            val_time.append(_epoch0_time)
            best_validate_loss = _epoch0_metrics[0]
            his_loss.append(_epoch0_metrics[0])
            trainloss_record.append(
                [np.nan, np.nan, np.nan, np.nan] + _epoch0_metrics
            )
            log_in_train_details.append(
                [0, np.nan, np.nan, np.nan, np.nan]
                + _epoch0_metrics + [0.0]
            )
            torch.save(engine.model.state_dict(), save_path + 'best.pth')
            _anchor_label = (
                '[StructuredWidthPrune]'
                if _structured_width_prune_finetune
                else (
                    '[LowRankStudentFT]'
                    if _retrieval_aware_head_finetune
                    else '[FrozenParent]'
                )
            )
            logger.info(
                '%s Epoch 000 anchor, Valid Loss: %.4f, '
                'Valid MAE: %.4f, Valid MAPE: %.4f, Valid RMSE: %.4f, '
                'Inference Time: %.4f secs',
                _anchor_label, *_epoch0_metrics, _epoch0_time,
            )

        for i in range(1, args.epochs + 1):
            train_loss = []
            train_mae = []
            train_mape = []
            train_rmse = []
            dataloader['train_loader'].shuffle()
            t1 = time.time()

            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                (
                    x, x_time, analog_retrieval,
                    exact_retrieval, exact_available,
                    candidate_future, candidate_available,
                    candidate_features,
                ) = unpack_dataloader_input(x)
                y, y_time = y
                trainx = torch.Tensor(x).to(device)
                trainy = torch.Tensor(y).to(device)
                trainxtime = torch.LongTensor(x_time).to(device)
                trainytime = torch.LongTensor(y_time).to(device)
                # 3-tuple: (x, x_time, y_time)；模型 forward 兼容 2/3-tuple
                trainx = [trainx, trainxtime, trainytime]
                if analog_retrieval is not None:
                    trainx.append(
                        torch.as_tensor(
                            analog_retrieval,
                            dtype=(
                                torch.float16
                                if analog_retrieval.dtype == np.float16
                                else torch.float32
                            ),
                            device=device,
                        )
                    )
                if exact_retrieval is not None:
                    cache_float_dtype = (
                        torch.float16
                        if exact_retrieval.dtype == np.float16
                        else torch.float32
                    )
                    trainx.extend([
                        torch.as_tensor(
                            exact_retrieval,
                            dtype=cache_float_dtype,
                            device=device,
                        ),
                        torch.as_tensor(
                            exact_available,
                            dtype=cache_float_dtype,
                            device=device,
                        ),
                    ])
                if candidate_future is not None:
                    trainx.extend([
                        torch.as_tensor(
                            candidate_future,
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            candidate_available,
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            candidate_features,
                            dtype=torch.float32,
                            device=device,
                        ),
                    ])
                metrics = engine.train(
                    trainx, trainy, batches_seen,
                    teacher_pred=(
                        dataloader['train_loader'].current_teacher_batch
                    ),
                    temporal_position=(
                        dataloader['train_loader']
                        .current_temporal_position_batch
                    ),
                )
                batches_seen += 1

                # ===== EMA 权重更新（每个 batch 后；SWA 改为每 epoch 收集，见下）=====
                if ema_shadow is not None:
                    for n, p in engine.model.named_parameters():
                        ema_shadow[n].mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)

                train_loss.append(metrics[0])
                train_mae.append(metrics[1])
                train_mape.append(metrics[2])
                train_rmse.append(metrics[3])
                if iter % args.print_every == 0:
                    log = 'Iter: {:03d} [{:d}], Train Loss: {:.4f},Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    logger.info(log.format(
                        iter, batches_seen, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]))
            t2 = time.time()
            train_time.append(t2 - t1)

            if args.use_swa and i >= swa_start_epoch:
                swa_model.update_parameters(engine.model)
            
            # ===== 验证前：若启用 EMA/SWA 且已激活，临时加载平均权重用于验证 =====
            _eval_backup_sd = None
            if args.use_swa and i >= swa_start_epoch:
                _eval_backup_sd = {k: v.detach().clone() for k, v in engine.model.state_dict().items()}
                engine.model.load_state_dict(swa_model.module.state_dict())
            elif ema_shadow is not None:
                _eval_backup_sd = {k: v.detach().clone() for k, v in engine.model.state_dict().items()}
                _ema_sd = {n: (ema_shadow[n] if n in ema_shadow else v)
                           for n, v in engine.model.state_dict().items()}
                engine.model.load_state_dict(_ema_sd)

            # validation
            valid_totals = {
                metric: np.zeros(args.predict_length, dtype=np.float64)
                for metric in ('loss', 'mae', 'mape', 'mse')
            }
            valid_counts = {
                metric: np.zeros(args.predict_length, dtype=np.float64)
                for metric in ('loss', 'mae', 'mape', 'rmse')
            }

            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                (
                    x, x_time, analog_retrieval,
                    exact_retrieval, exact_available,
                    candidate_future, candidate_available,
                    candidate_features,
                ) = unpack_dataloader_input(x)
                y, y_time = y
                valx = torch.Tensor(x).to(device)
                valy = torch.Tensor(y).to(device)
                valxtime = torch.LongTensor(x_time).to(device)
                valytime = torch.LongTensor(y_time).to(device)
                valx = [valx, valxtime, valytime]
                if analog_retrieval is not None:
                    valx.append(
                        torch.as_tensor(
                            analog_retrieval,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                if exact_retrieval is not None:
                    valx.extend([
                        torch.as_tensor(
                            exact_retrieval,
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            exact_available,
                            dtype=torch.float32,
                            device=device,
                        ),
                    ])
                if candidate_future is not None:
                    valx.extend([
                        torch.as_tensor(
                            candidate_future,
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            candidate_available,
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.as_tensor(
                            candidate_features,
                            dtype=torch.float32,
                            device=device,
                        ),
                    ])

                metrics = engine.eval(valx, valy, return_counts=True)
                horizon = metrics[4]['horizon']
                loss_values = np.asarray(horizon['loss'])
                mae_values = np.asarray(horizon['mae'])
                mape_values = np.asarray(horizon['mape'])
                rmse_values = np.asarray(horizon['rmse'])
                valid_totals['loss'] += (
                    loss_values * np.asarray(horizon['loss_count'])
                )
                valid_totals['mae'] += (
                    mae_values * np.asarray(horizon['mae_count'])
                )
                valid_totals['mape'] += (
                    mape_values * np.asarray(horizon['mape_count'])
                )
                valid_totals['mse'] += (
                    rmse_values ** 2 * np.asarray(horizon['rmse_count'])
                )
                for key in valid_counts:
                    valid_counts[key] += np.asarray(horizon[f'{key}_count'])

            # 验证结束：恢复原始权重，继续训练
            if _eval_backup_sd is not None:
                engine.model.load_state_dict(_eval_backup_sd)

            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            logger.info(log.format(i, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mae = np.mean(train_mae)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            horizon_loss = valid_totals['loss'] / np.maximum(valid_counts['loss'], 1.0)
            horizon_mae = valid_totals['mae'] / np.maximum(valid_counts['mae'], 1.0)
            horizon_mape = valid_totals['mape'] / np.maximum(valid_counts['mape'], 1.0)
            horizon_rmse = np.sqrt(
                valid_totals['mse'] / np.maximum(valid_counts['rmse'], 1.0)
            )
            mvalid_loss = float(horizon_loss.mean())
            mvalid_mae = float(horizon_mae.mean())
            mvalid_mape = float(horizon_mape.mean())
            mvalid_rmse = float(horizon_rmse.mean())

            if engine.lr_scheduler is not None:
                if engine.lr_scheduler_type.lower() == 'reducelronplateau':
                    engine.lr_scheduler.step(mvalid_loss)
                elif engine.lr_scheduler_type.lower() == 'cosinelr':
                    engine.lr_scheduler.step(i)
                else:
                    engine.lr_scheduler.step()

            his_loss.append(mvalid_loss)
            trainloss_record.append(
                [mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse,
                 mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse])
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
            log += 'Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            logger.info(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                                mvalid_rmse, (t2 - t1)))
            log_in_train_details.append([i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                                        mvalid_rmse, (t2 - t1)])

            if best_validate_loss > mvalid_loss:
                best_validate_loss = mvalid_loss
                validate_score_non_decrease_count = 0
                # 保存：EMA/SWA 启用且已激活时存平均权重，否则存原始权重
                if args.use_swa and i >= swa_start_epoch:
                    sd_to_save = swa_model.module.state_dict()
                    _save_tag = "SWA"
                elif args.use_ema:
                    sd_to_save = {n: (ema_shadow[n] if n in ema_shadow else v)
                                  for n, v in engine.model.state_dict().items()}
                    _save_tag = "EMA"
                else:
                    sd_to_save = engine.model.state_dict()
                    _save_tag = "raw"
                torch.save(sd_to_save, save_path + "best.pth")
                logger.info('got best validation result [{} weights]: {:.4f}, {:.4f}, {:.4f}'.format(
                    _save_tag, mvalid_loss, mvalid_mape, mvalid_rmse))
                logger.info(f'---- {expid}_{args.modelid}_{args.dataset_name}_{args.note} ----')
            
            else:
                if i >= args.early_stop_start_epoch:
                    validate_score_non_decrease_count += 1

            if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
                break
        
        avg_train_time = np.mean(train_time)
        avg_inference_time = np.mean(val_time)
        logger.info(
            "Average Training Time: {:.4f} secs/epoch".format(avg_train_time))
        logger.info("Average Inference Time: {:.4f} secs".format(avg_inference_time))

        training_time = (time.time() - all_start_time) / 60

        bestid = np.argmin(his_loss)
        logger.info("Training finished")
        logger.info(
            f"The valid loss on best model is {str(round(his_loss[bestid], 4))}")
        train_metric = trainloss_record[bestid][0:4]
        valid_metric = trainloss_record[bestid][4:]
    else:
        logger.info(f'Loaded model from {save_path}best.pth')
        train_metric = []
        valid_metric = []
        log_in_train_details = []

    if args.validation_only:
        if bool(args.save_validation_output):
            engine.model.load_state_dict(
                torch.load(save_path + "best.pth", map_location=device)
            )
            engine.model.eval()
            validation_predictions = []
            validation_truths = []
            validation_risk_scores = []
            elastic_predictor = engine.model.time_wise_predictor
            export_elastic_subnets = bool(getattr(
                args, 'use_elastic_rank_router', 0
            ))
            elastic_low_predictions = []
            elastic_high_predictions = []
            elastic_probabilities = []
            elastic_hard_gates = []
            with torch.no_grad():
                for x, y in dataloader['val_loader'].get_iterator():
                    (
                        x, x_time, analog_retrieval,
                        exact_retrieval, exact_available,
                        candidate_future, candidate_available,
                        candidate_features,
                    ) = unpack_dataloader_input(x)
                    y, y_time = y
                    model_input = [
                        torch.as_tensor(x, dtype=torch.float32, device=device),
                        torch.as_tensor(x_time, dtype=torch.long, device=device),
                        torch.as_tensor(y_time, dtype=torch.long, device=device),
                    ]
                    if analog_retrieval is not None:
                        model_input.append(torch.as_tensor(
                            analog_retrieval,
                            dtype=torch.float32,
                            device=device,
                        ))
                    if exact_retrieval is not None:
                        model_input.extend([
                            torch.as_tensor(
                                exact_retrieval,
                                dtype=torch.float32,
                                device=device,
                            ),
                            torch.as_tensor(
                                exact_available,
                                dtype=torch.float32,
                                device=device,
                            ),
                        ])
                    if candidate_future is not None:
                        model_input.extend([
                            torch.as_tensor(
                                candidate_future,
                                dtype=torch.float32,
                                device=device,
                            ),
                            torch.as_tensor(
                                candidate_available,
                                dtype=torch.float32,
                                device=device,
                            ),
                            torch.as_tensor(
                                candidate_features,
                                dtype=torch.float32,
                                device=device,
                            ),
                        ])
                    # trainer._forward already applies inverse_transform when
                    # inverse=True, matching the validation metric path.
                    prediction = engine._forward(model_input)
                    validation_predictions.append(prediction.detach().cpu())
                    if export_elastic_subnets:
                        elastic_probabilities.append(
                            elastic_predictor.last_elastic_probability.detach().cpu()
                        )
                        elastic_hard_gates.append(
                            elastic_predictor.last_elastic_hard_gate.detach().cpu()
                        )
                        try:
                            elastic_predictor.elastic_force_rank = (
                                elastic_predictor.elastic_low_rank
                            )
                            low_prediction = engine._forward(model_input)
                            elastic_predictor.elastic_force_rank = (
                                elastic_predictor.factor_rank
                            )
                            high_prediction = engine._forward(model_input)
                        finally:
                            elastic_predictor.elastic_force_rank = None
                        elastic_low_predictions.append(
                            low_prediction.detach().cpu()
                        )
                        elastic_high_predictions.append(
                            high_prediction.detach().cpu()
                        )
                    if getattr(engine.model, 'last_risk_score', None) is not None:
                        validation_risk_scores.append(
                            engine.model.last_risk_score.detach().cpu()
                        )
                    validation_truths.append(torch.as_tensor(
                        y, dtype=torch.float32
                    ))
            validation_prediction = torch.cat(
                validation_predictions, dim=0
            ).numpy()
            validation_truth = torch.cat(validation_truths, dim=0).numpy()
            validation_output_path = os.path.join(
                save_path, 'validation_output.npz'
            )
            validation_payload = {
                'validation_prediction': validation_prediction,
                'validation_truth': validation_truth,
                'prediction_scale': np.asarray('raw'),
                'split': np.asarray('validation'),
            }
            if validation_risk_scores:
                validation_payload['validation_risk_score'] = torch.cat(
                    validation_risk_scores, dim=0
                ).numpy()
            if export_elastic_subnets:
                validation_payload.update({
                    'elastic_low_prediction': torch.cat(
                        elastic_low_predictions, dim=0
                    ).numpy(),
                    'elastic_high_prediction': torch.cat(
                        elastic_high_predictions, dim=0
                    ).numpy(),
                    'elastic_router_probability': torch.cat(
                        elastic_probabilities, dim=0
                    ).numpy(),
                    'elastic_hard_gate': torch.cat(
                        elastic_hard_gates, dim=0
                    ).numpy(),
                })
            np.savez_compressed(validation_output_path, **validation_payload)
            logger.info(
                'Raw-scale validation output saved to %s; Test was not read.',
                validation_output_path,
            )
        logger.info(
            'Validation-only run complete; Test evaluation intentionally skipped.'
        )
        sys.exit(0)

    # testing
    engine.model.load_state_dict(
        torch.load(save_path + "best.pth", map_location=device)
    )
    engine.model.eval()
    if hasattr(engine.model, 'selective_lag_gates'):
        with torch.no_grad():
            raw_gates = engine.model.selective_lag_gates.detach().cpu()
            effective_gates = torch.tanh(raw_gates)
        logger.info(
            '[SelectiveTLE] best raw gates (references 1..T-1): %s',
            [round(float(value), 6) for value in raw_gates]
        )
        logger.info(
            '[SelectiveTLE] best tanh gates (references 1..T-1): %s',
            [round(float(value), 6) for value in effective_gates]
        )
    if hasattr(engine.model, 'node_linear_dynamics'):
        with torch.no_grad():
            dynamics = engine.model.node_linear_dynamics
            effective_weight = dynamics.scale * dynamics.weight.detach()
            horizon_rms = effective_weight.square().mean(
                dim=(0, 1, 3)
            ).sqrt().cpu()
            nonzero_fraction = (
                effective_weight.ne(0).float().mean().cpu().item()
            )
        logger.info(
            '[NodeLinearDynamics] scale=%.4f, nonzero_fraction=%.6f, '
            'effective_weight_horizon_rms=%s',
            dynamics.scale,
            nonzero_fraction,
            [round(float(value), 6) for value in horizon_rms],
        )
    if hasattr(engine.model, 'regime_gated_residual'):
        with torch.no_grad():
            adapter = engine.model.regime_gated_residual
            output_layer = adapter.correction_net[-1]
            output_values = torch.cat([
                output_layer.weight.detach().reshape(-1),
                output_layer.bias.detach().reshape(-1),
            ])
            nonzero_fraction = (
                output_values.ne(0).float().mean().cpu().item()
            )
            output_rms = output_values.square().mean().sqrt().cpu().item()
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
        logger.info(
            '[RegimeGatedResidual] scale=%.4f, parameters=%d, '
            'output_nonzero_fraction=%.6f, output_rms=%.8f',
            adapter.scale, adapter_parameters,
            nonzero_fraction, output_rms,
        )
    if hasattr(engine.model, 'multires_block_residual'):
        with torch.no_grad():
            adapter = engine.model.multires_block_residual
            output_values = torch.cat([
                adapter.block_projection.weight.detach().reshape(-1),
                adapter.block_projection.bias.detach().reshape(-1),
            ])
            nonzero_fraction = (
                output_values.ne(0).float().mean().cpu().item()
            )
            output_rms = output_values.square().mean().sqrt().cpu().item()
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
        logger.info(
            '[MultiResolutionBlockResidual] block_size=%d, scale=%.4f, '
            'parameters=%d, output_nonzero_fraction=%.6f, '
            'output_rms=%.8f',
            adapter.block_size, adapter.scale, adapter_parameters,
            nonzero_fraction, output_rms,
        )
    if hasattr(engine.model, 'seasonal_drift_residual'):
        with torch.no_grad():
            adapter = engine.model.seasonal_drift_residual
            horizon_scale = adapter.max_scale * torch.tanh(
                adapter.horizon_logits.detach()
            )
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
        logger.info(
            '[SeasonalDriftResidual] parameters=%d, max_scale=%.4f, '
            'horizon_scales=%s',
            adapter_parameters,
            adapter.max_scale,
            [round(float(value), 6) for value in horizon_scale.cpu()],
        )
    if hasattr(engine.model, 'analog_retrieval_residual'):
        with torch.no_grad():
            adapter = engine.model.analog_retrieval_residual
            horizon_blend = adapter.max_blend * torch.tanh(
                adapter.horizon_logits.detach()
            )
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
        logger.info(
            '[AnalogRetrievalResidual] parameters=%d, K=%d, '
            'max_blend=%.4f, horizon_blends=%s',
            adapter_parameters,
            args.analog_retrieval_k,
            adapter.max_blend,
            [round(float(value), 6) for value in horizon_blend.cpu()],
        )
    if hasattr(engine.model, 'analog_lowrank_retrieval_residual'):
        with torch.no_grad():
            adapter = engine.model.analog_lowrank_retrieval_residual
            lowrank_logits = torch.einsum(
                'nr,hr->hn',
                adapter.node_factor.detach(),
                adapter.horizon_factor.detach(),
            )
            blend = adapter.max_blend * torch.tanh(
                adapter.horizon_logits.detach()[:, None] + lowrank_logits
            )
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
            horizon_mean = blend.mean(dim=1)
            horizon_min = blend.amin(dim=1)
            horizon_max = blend.amax(dim=1)
        logger.info(
            '[AnalogLowRankRetrievalResidual] parameters=%d, rank=%d, K=%d, '
            'max_blend=%.4f, horizon_mean=%s, horizon_min=%s, '
            'horizon_max=%s',
            adapter_parameters,
            adapter.rank,
            args.analog_retrieval_k,
            adapter.max_blend,
            [round(float(value), 6) for value in horizon_mean.cpu()],
            [round(float(value), 6) for value in horizon_min.cpu()],
            [round(float(value), 6) for value in horizon_max.cpu()],
        )
    if hasattr(engine.model, 'analog_temporal_lowrank_decoder'):
        with torch.no_grad():
            adapter = engine.model.analog_temporal_lowrank_decoder
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
            encoder_rms = (
                adapter.encoder.detach().square().mean().sqrt().cpu().item()
            )
            decoder_rms = (
                adapter.decoder.detach().square().mean().sqrt().cpu().item()
            )
        logger.info(
            '[AnalogTemporalLowRankDecoder] parameters=%d, rank=%d, '
            'max_scale=%.4f, encoder_rms=%.8f, decoder_rms=%.8f',
            adapter_parameters,
            adapter.rank,
            adapter.max_scale,
            encoder_rms,
            decoder_rms,
        )
    if hasattr(engine.model, 'analog_temporal_gap_residual'):
        with torch.no_grad():
            adapter = engine.model.analog_temporal_gap_residual
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
            encoder_rms = (
                adapter.encoder.detach().square().mean().sqrt().cpu().item()
            )
            decoder_rms = (
                adapter.decoder.detach().square().mean().sqrt().cpu().item()
            )
            node_gain = 1.0 + torch.tanh(adapter.node_factor.detach())
        logger.info(
            '[AnalogTemporalGapResidual] parameters=%d, rank=%d, '
            'max_scale=%.4f, encoder_rms=%.8f, decoder_rms=%.8f, '
            'node_gain_min=%.6f, node_gain_mean=%.6f, node_gain_max=%.6f',
            adapter_parameters,
            adapter.rank,
            adapter.max_scale,
            encoder_rms,
            decoder_rms,
            float(node_gain.min().cpu()),
            float(node_gain.mean().cpu()),
            float(node_gain.max().cpu()),
        )
    if hasattr(
        engine.model, 'analog_node_horizon_retrieval_residual'
    ):
        with torch.no_grad():
            adapter = (
                engine.model.analog_node_horizon_retrieval_residual
            )
            blend = adapter.max_blend * torch.tanh(
                adapter.node_horizon_logits.detach()
            )
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
            horizon_mean = blend.mean(dim=1)
            horizon_min = blend.amin(dim=1)
            horizon_max = blend.amax(dim=1)
        logger.info(
            '[AnalogNodeHorizonRetrievalResidual] parameters=%d, K=%d, '
            'aggregation=%s, max_blend=%.4f, horizon_mean=%s, '
            'horizon_min=%s, horizon_max=%s',
            adapter_parameters,
            args.analog_retrieval_k,
            args.analog_retrieval_aggregation,
            adapter.max_blend,
            [round(float(value), 6) for value in horizon_mean.cpu()],
            [round(float(value), 6) for value in horizon_min.cpu()],
            [round(float(value), 6) for value in horizon_max.cpu()],
        )
    if hasattr(
        engine.model, 'analog_confidence_aware_retrieval_residual'
    ):
        with torch.no_grad():
            adapter = (
                engine.model
                .analog_confidence_aware_retrieval_residual
            )
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
            weight_rms = (
                adapter.quality_weight.detach().square().mean().sqrt()
                .cpu().item()
            )
            horizon_bias = adapter.horizon_bias.detach().cpu()
        logger.info(
            '[AnalogConfidenceAwareRetrievalResidual] parameters=%d, '
            'max_adjustment=%.4f, quality_weight_rms=%.8f, '
            'horizon_bias=%s',
            adapter_parameters,
            adapter.max_adjustment,
            weight_rms,
            [round(float(value), 6) for value in horizon_bias],
        )
    if hasattr(engine.model, 'analog_weekly_retrieval_residual'):
        with torch.no_grad():
            adapter = engine.model.analog_weekly_retrieval_residual
            lowrank_logits = torch.einsum(
                'nr,hr->hn',
                adapter.node_factor.detach(),
                adapter.horizon_factor.detach(),
            )
            blend = adapter.max_blend * torch.tanh(
                adapter.horizon_logits.detach()[:, None] + lowrank_logits
            )
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
        logger.info(
            '[AnalogWeeklyRetrievalResidual] parameters=%d, rank=%d, K=%d, '
            'max_blend=%.4f, horizon_mean=%s, horizon_min=%s, '
            'horizon_max=%s',
            adapter_parameters,
            adapter.rank,
            args.analog_weekly_retrieval_k,
            adapter.max_blend,
            [
                round(float(value), 6)
                for value in blend.mean(dim=1).cpu()
            ],
            [
                round(float(value), 6)
                for value in blend.amin(dim=1).cpu()
            ],
            [
                round(float(value), 6)
                for value in blend.amax(dim=1).cpu()
            ],
        )
    if hasattr(engine.model, 'analog_metric_retrieval_residual'):
        with torch.no_grad():
            adapter = engine.model.analog_metric_retrieval_residual
            lowrank_logits = torch.einsum(
                'nr,hr->hn',
                adapter.node_factor.detach(),
                adapter.horizon_factor.detach(),
            )
            blend = adapter.max_blend * torch.tanh(
                adapter.horizon_logits.detach()[:, None] + lowrank_logits
            )
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
        logger.info(
            '[AnalogMetricRetrievalResidual] parameters=%d, rank=%d, K=%d, '
            'max_blend=%.4f, horizon_mean=%s, horizon_min=%s, '
            'horizon_max=%s',
            adapter_parameters,
            adapter.rank,
            args.analog_metric_retrieval_k,
            adapter.max_blend,
            [
                round(float(value), 6)
                for value in blend.mean(dim=1).cpu()
            ],
            [
                round(float(value), 6)
                for value in blend.amin(dim=1).cpu()
            ],
            [
                round(float(value), 6)
                for value in blend.amax(dim=1).cpu()
            ],
        )
    if hasattr(engine.model, 'analog_exact_match_residual'):
        with torch.no_grad():
            adapter = engine.model.analog_exact_match_residual
            horizon_blend = adapter.max_blend * torch.tanh(
                adapter.horizon_logits.detach()
            )
            adapter_parameters = sum(
                parameter.numel() for parameter in adapter.parameters()
            )
        logger.info(
            '[AnalogExactMatchResidual] parameters=%d, max_blend=%.4f, '
            'horizon_blends=%s',
            adapter_parameters,
            adapter.max_blend,
            [round(float(value), 6) for value in horizon_blend.cpu()],
        )

    @torch.no_grad()
    def collect_predictions(loader, num_samples=None):
        outputs, truths, last_values = [], [], []
        delta_scales = []
        elapsed = 0.0
        for x_batch, y_batch in loader.get_iterator():
            (
                x_batch, x_time, analog_retrieval,
                exact_retrieval, exact_available,
                candidate_future, candidate_available,
                candidate_features,
            ) = unpack_dataloader_input(x_batch)
            y_batch, y_time = y_batch
            model_x = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            model_input = [
                model_x,
                torch.as_tensor(x_time, dtype=torch.long, device=device),
                torch.as_tensor(y_time, dtype=torch.long, device=device),
            ]
            if analog_retrieval is not None:
                model_input.append(
                    torch.as_tensor(
                        analog_retrieval,
                        dtype=torch.float32,
                        device=device,
                    )
                )
            if exact_retrieval is not None:
                model_input.extend([
                    torch.as_tensor(
                        exact_retrieval,
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        exact_available,
                        dtype=torch.float32,
                        device=device,
                    ),
                ])
            if candidate_future is not None:
                model_input.extend([
                    torch.as_tensor(
                        candidate_future,
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        candidate_available,
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        candidate_features,
                        dtype=torch.float32,
                        device=device,
                    ),
                ])
            start = time.time()
            prediction, _ = engine.model(model_input)
            elapsed += time.time() - start
            outputs.append(prediction.detach().cpu())
            if hasattr(engine.model, 'volatility_delta_scaler'):
                current_scale = engine.model.volatility_delta_scaler.last_scale
                if current_scale is not None:
                    delta_scales.append(current_scale.cpu())
            truths.append(
                torch.as_tensor(y_batch, dtype=torch.float32).clone()
            )
            last_values.append(
                scaler.inverse_transform(model_x[:, -1]).detach().cpu()
            )

        prediction = torch.cat(outputs, dim=0)
        truth = torch.cat(truths, dim=0)
        last_value = torch.cat(last_values, dim=0)
        if num_samples is not None:
            prediction = prediction[:num_samples]
            truth = truth[:num_samples]
            last_value = last_value[:num_samples]
        scale_summary = None
        if delta_scales:
            scale_values = torch.cat(delta_scales, dim=0)
            if num_samples is not None:
                scale_values = scale_values[:num_samples]
            reduce_dims = (0, 2, 3)
            scale_summary = {
                'mean': scale_values.mean(dim=reduce_dims).numpy(),
                'min': scale_values.amin(dim=reduce_dims).numpy(),
                'max': scale_values.amax(dim=reduce_dims).numpy(),
            }
        return prediction, truth, last_value, elapsed, scale_summary

    (
        yhat_normalized, realy, test_last_value, inference_time,
        test_delta_scale_summary,
    ) = collect_predictions(dataloader['test_loader'], num_test_samples)
    if test_delta_scale_summary is not None:
        for statistic, values in test_delta_scale_summary.items():
            logger.info(
                '[VolatilityDeltaScale] test per-horizon %s: %s',
                statistic,
                [round(float(value), 6) for value in values],
            )
    yhat = scaler.inverse_transform(yhat_normalized) if inverse else yhat_normalized
    all_amae, all_amape, all_armse = [], [], [] # all results of single time step
    MAE_list, MAPE_list, RMSE_list = [], [], [] # average results of all time steps 
    for feature_idx in range(args.pre_dim):
        amae, amape, armse = [], [], []
        pred_feature = yhat[..., feature_idx]
        real_feature = realy[..., feature_idx]
        for i in range(args.predict_length):
            pred = pred_feature[:, i]
            real = real_feature[:, i]
            
            metrics = utils.metric_torch(pred, real, mask_val=args.eval_mask)
            log = 'Evaluate best model on test data for [dim{:d}] horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            logger.info(log.format(feature_idx, i + 1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = '[dim{:d}] On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        MAE = np.mean(amae)
        MAPE = np.mean(amape)
        RMSE = np.mean(armse)
        logger.info(log.format(feature_idx, args.predict_length, MAE, MAPE, RMSE))
        MAE_list.append(MAE)
        MAPE_list.append(MAPE)
        RMSE_list.append(RMSE)
        all_amae.append(amae)
        all_amape.append(amape)
        all_armse.append(armse)

    if not os.path.exists(f'./EXP_results/{args.modelid}/'):
        os.makedirs(f'./EXP_results/{args.modelid}/')
    result_csv_path = f'./EXP_results/{args.modelid}/{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_Results.csv'

    with open(result_csv_path, 'a+', newline='')as f0:
        f_csv = csv.writer(f0)
        row = [expid, args.dataset_name, args.note,args.M,args.hid_dim, args.n_heads,'test']
        if args.pre_dim >1:
            row.extend([f'avg', np.array(MAE_list).mean(),np.array(MAPE_list).mean()*100,np.array(RMSE_list).mean()])
        for feature_idx in range(args.pre_dim):
            row.extend([f'dim-{feature_idx}',MAE_list[feature_idx], MAPE_list[feature_idx]
                       * 100, RMSE_list[feature_idx]])
        row.extend(['val'])
        row.extend(valid_metric)
        row.extend(['train'])
        row.extend(train_metric)
        f_csv.writerow(row)

    logger.info('log and results saved.')

    if bool(args.save_output):
        output_path = f'{save_path}/{args.modelid}_{args.dataset_name}_exp{expid}_output.npz'
        (
            val_prediction_normalized, val_truth, val_last_value, _,
            val_delta_scale_summary,
        ) = collect_predictions(dataloader['val_loader'])
        val_prediction = (
            scaler.inverse_transform(val_prediction_normalized)
            if inverse else val_prediction_normalized
        )
        output_values = dict(
            prediction=yhat.cpu().numpy(),
            truth=realy.cpu().numpy(),
            last_value=test_last_value.cpu().numpy(),
            validation_prediction=val_prediction.cpu().numpy(),
            validation_truth=val_truth.cpu().numpy(),
            validation_last_value=val_last_value.cpu().numpy(),
            prediction_scale=np.asarray('raw'),
            scaler_mean=np.asarray(scaler.mean),
            scaler_std=np.asarray(scaler.std),
        )
        if test_delta_scale_summary is not None:
            for statistic, values in test_delta_scale_summary.items():
                output_values[f'volatility_delta_scale_test_{statistic}'] = values
        if val_delta_scale_summary is not None:
            for statistic, values in val_delta_scale_summary.items():
                output_values[f'volatility_delta_scale_validation_{statistic}'] = values
        np.savez_compressed(output_path, **output_values)
        logger.info('output npz saved with raw-scale validation/test predictions.')

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
