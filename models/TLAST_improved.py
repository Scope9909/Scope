import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from math import log, sqrt

from functools import partial
from einops import rearrange, repeat
from layers.DIFFormer import DIFFormerConv
from layers.ProbAttention import ProbAttention
from layers.GraphConv import AdaptiveGraphConv, MultiScaleGraphConv



# ========================= 原有类定义（从 TLAST.py 复制）=========================
# 包含：MultiHeadsAttention, AttentionLayer, DIFFormer_EncoderLayer, 
#       Full_Prob_EncoderLayer, EncoderLayer, time_wise_predictor,
#       getTimeLagEmbedding, getTimeEmbedding, getSpatialEmbedding, DataEncoding
# ================================================================================

class MultiHeadsAttention(nn.Module):
    """
    The Attention operation
    """
    def __init__(self, scale=None, attention_dropout=0.1, returnA=False):
        super(MultiHeadsAttention, self).__init__()
        self.scale = scale
        self.returnA = returnA
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.returnA:
            return V.contiguous(), A.contiguous()
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1, returnA=False, att_type='full',prob_factor=5):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (hid_dim//n_heads)
        d_values = d_values or (hid_dim//n_heads)

        if att_type == 'full' or att_type == 'proxy':
            self.inner_attention = MultiHeadsAttention(
                scale=None, attention_dropout=dropout, returnA=returnA)
        elif att_type=='prob':
            self.inner_attention = ProbAttention(False, prob_factor, attention_dropout=dropout, output_attention=returnA)
        self.query_projection = nn.Linear(hid_dim, d_keys * n_heads)
        self.key_projection = nn.Linear(hid_dim, d_keys * n_heads)
        self.value_projection = nn.Linear(hid_dim, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, hid_dim)
        self.n_heads = n_heads
        self.returnA = returnA
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, A = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        if self.returnA:
            return out, A
        else:
            return out, None


class DIFFormer_EncoderLayer(nn.Module):
    """
    Use Difformer Attention
    """

    def __init__(self, hid_dim, n_heads, norm_flag,  
                 query_cross_time=True,
                 d_ff=None, dropout=0.1,return_att=False, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.return_att = return_att
        self.att_layer = DIFFormerConv(hid_dim, hid_dim, num_heads=n_heads,output_attn=return_att)
        self.dropout = nn.Dropout(dropout)
        self.norm_flag=norm_flag
        assert norm_flag in ['none', 'first', 'after']
        self.query_cross_time = query_cross_time
        if norm_flag == 'first' and query_cross_time: 
            self.norm0 = nn.LayerNorm(hid_dim)
        if norm_flag != 'none':
            self.norm1 = nn.LayerNorm(hid_dim)
            self.norm2 = nn.LayerNorm(hid_dim)
        assert activation in ['gelu','relu','GLU']
        activation_func_dict={'gelu':nn.GELU(),
                              'relu':nn.ReLU(),
                              'GLU':nn.GLU()}
        if activation=='GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP1 = nn.Sequential(nn.Linear(hid_dim, d_ff1),
                                  activation_func_dict[activation],
                                  nn.Linear(d_ff, hid_dim))

    def forward(self, data):  # data:BTNC
        batch = data.shape[0]
        T = data.shape[1]
        
        if self.query_cross_time:
            now = data[:, -1, :, :]
            query = repeat(now, 'b n c -> (b t) n c', t = T)
            data = rearrange(data, 'b t n c-> (b t) n c')
            if self.norm_flag=='first': 
                query=self.norm0(query)     
                x = self.norm1(data)
            else:
                x = data
            x, A = self.att_layer(query, x)
        else:
            data = rearrange(data, 'b t n c-> (b t) n c')
            if self.norm_flag=='first': 
                x = self.norm1(data)
            else: 
                x = data
            x, A = self.att_layer(x, x)
        
        x = data + self.dropout(x)
        if self.norm_flag == 'first': x = self.norm2(x)
        elif self.norm_flag == 'after': x = self.norm1(x)
        x = x + self.dropout(self.MLP1(x))
        if self.norm_flag=='after': x = self.norm2(x)

        final_out = rearrange(x, '(b T) N d -> b T N d', b=batch)
        if self.return_att:
            A = rearrange(A, '(b t) h l s -> b t h l s', b=batch)
            return final_out, A
        return final_out, None


class Full_Prob_EncoderLayer(nn.Module):
    """
    Use FullAttention or ProbAttention
    """

    def __init__(self, hid_dim, n_heads, att_type, norm_flag='first', d_ff=None, dropout=0.1, att_dropout=0.1, return_att=False, activation='gelu', query_cross_time=True, prob_factor=None):
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.return_att = return_att
        self.query_cross_time = query_cross_time
        self.att_layer = AttentionLayer(hid_dim, n_heads, dropout=att_dropout, att_type=att_type, returnA=return_att,prob_factor=prob_factor)
        self.dropout = nn.Dropout(dropout)
        self.norm_flag = norm_flag
        
        assert norm_flag in ['none', 'first', 'after']
        if norm_flag=='first' and query_cross_time: 
            self.norm0 = nn.LayerNorm(hid_dim)
        if norm_flag != 'none':
            self.norm1 = nn.LayerNorm(hid_dim)
            self.norm2 = nn.LayerNorm(hid_dim)

        assert activation in ['gelu','relu','GLU']
        activation_func_dict={'gelu':nn.GELU(),
                              'relu':nn.ReLU(),
                              'GLU':nn.GLU()}
        if activation=='GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP1 = nn.Sequential(nn.Linear(hid_dim, d_ff1),
                                  activation_func_dict[activation],
                                  nn.Linear(d_ff, hid_dim))

    def forward(self, data):  # data:BTNC
        batch = data.shape[0]
        T = data.shape[1]
        
        if self.query_cross_time:
            now = data[:, -1, :, :]
            query = repeat(now, 'b n c -> (b t) n c', t = T)               
            data = rearrange(data, 'b t n c-> (b t) n c')
            if self.norm_flag=='first': 
                query=self.norm0(query)     
                x = self.norm1(data)
            else: 
                x = data
            x, A = self.att_layer(query, x, x)

        else:
            data = rearrange(data, 'b t n c-> (b t) n c')
            if self.norm_flag=='first': 
                x = self.norm1(data)
            else: 
                x = data
            x, A = self.att_layer(x, x, x)
        
        x = data + self.dropout(x)
        if self.norm_flag == 'first': x = self.norm2(x)
        elif self.norm_flag == 'after': x = self.norm1(x)
        x = x + self.dropout(self.MLP1(x))
        if self.norm_flag=='after': x = self.norm2(x)

        final_out = rearrange(x, '(b T) N d -> b T N d', b=batch)
        if self.return_att:
            A = rearrange(A, '(b t) h l s -> b t h l s', b=batch)
            return final_out, A
        return final_out, None


class EncoderLayer(nn.Module):
    """
    Use Proxy Attention (原始版本，保留以向后兼容)
    """

    def __init__(self, factor, hid_dim, n_heads, num_nodes=None,
                 d_ff=None, dropout=0.1, att_dropout=0.1,activation='gelu',
                   query_cross_time=True, norm_flag=1, dynamic_proxy=True, return_att=False):
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.query_cross_time= query_cross_time
        self.norm_flag = norm_flag
        self.return_att = return_att
        self.dynamic_proxy = dynamic_proxy
        assert num_nodes is not None
        if dynamic_proxy:
            self.readout_fc = nn.Linear(num_nodes, factor)
        else:
            self.proxy_tensor = nn.Parameter(torch.zeros(1, factor, hid_dim))
            nn.init.xavier_uniform_(self.proxy_tensor)
            
        self.node2proxy = AttentionLayer(hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att)
        self.proxy2node = AttentionLayer(hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att)

        self.dropout = nn.Dropout(dropout)
        

        if norm_flag=='first': 
            self.norm0 = nn.LayerNorm(hid_dim)
        if norm_flag != 'none':
            self.norm1 = nn.LayerNorm(hid_dim)
            self.norm2 = nn.LayerNorm(hid_dim)

        assert activation in ['gelu','relu','GLU']
        activation_func_dict={'gelu':nn.GELU(),
                              'relu':nn.ReLU(),
                              'GLU':nn.GLU()}
        if activation=='GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP2 = nn.Sequential(nn.Linear(hid_dim, d_ff1),
                                  activation_func_dict[activation],
                                  nn.Linear(d_ff, hid_dim))

    def forward(self, data):  # data:BTNC
        now = data[:, -1, :, :]
        batch = data.shape[0]
        T = data.shape[1]

        if self.dynamic_proxy:
            if self.query_cross_time:
                now = data[:, -1, :, :]
                temp = rearrange(now, 'b n c -> b c n')
                z_proxy= self.readout_fc(temp)
                z_proxy= repeat(z_proxy, 'b c k -> (b repeat) c k', repeat=T)
                z_proxy= rearrange(z_proxy, 'bt c K -> bt K c')
            else:
                temp = rearrange(data, 'b t n c -> b t c n')
                z_proxy= self.readout_fc(temp)
                z_proxy= rearrange(z_proxy, 'b t c K -> (b t) K c')     
        else:
            z_proxy = repeat(self.proxy_tensor, '1 m d ->(1 repeat) m d', repeat=T*batch)

        data = rearrange(data, 'b t n c-> (b t) n c')

        if self.norm_flag=='first':
            z_proxy = self.norm0(z_proxy)
            data = self.norm1(data)
            proxy_feature, A1 = self.node2proxy(z_proxy, data, data)      
            node_feature, A2 = self.proxy2node(data, proxy_feature, proxy_feature)
            enc_feature= data + self.dropout(node_feature)
            enc_feature = self.norm2(enc_feature)
            enc_feature= enc_feature+ self.dropout(self.MLP2(enc_feature))
        elif self.norm_flag == 'after':
            proxy_feature, A1 = self.node2proxy(z_proxy, data, data)
            node_feature, A2 = self.proxy2node(data, proxy_feature, proxy_feature)
            enc_feature= data + self.dropout(node_feature)
            enc_feature = self.norm1(enc_feature)
            enc_feature = enc_feature+ self.dropout(self.MLP2(enc_feature))
            enc_feature = self.norm2(enc_feature)
        elif self.norm_flag == 'none':
            proxy_feature, A1 = self.node2proxy(z_proxy, data, data)
            node_feature, A2 = self.proxy2node(data, proxy_feature, proxy_feature)
            enc_feature= data + self.dropout(node_feature)
            enc_feature= enc_feature+ self.dropout(self.MLP2(enc_feature))

        final_out = rearrange(
            enc_feature, '(b T) N hid_dim -> b T N hid_dim', b=batch)
        if self.return_att:
            A1 = rearrange(A1, '(b t) h l s -> b t h l s', b=batch)
            A2 = rearrange(A2, '(b t) h l s -> b t h l s', b=batch)
            return final_out, [A1, A2]
        else:
            return final_out, None


class time_wise_predictor(nn.Module):
    """
    原始Predictor：12个独立Linear，无依赖建模
    
    这是TLAST原始版本的Predictor，每个预测步独立预测，不建模预测步之间的依赖关系。
    """
    def __init__(self, num_nodes, input_length, predict_length, hid_dim, pre_dim):
        super(time_wise_predictor, self).__init__()
        self.predictors = nn.ModuleList([
            nn.Linear(input_length * hid_dim, pre_dim) for _ in range(predict_length)
        ])
        self.predict_length = predict_length
        self.input_length = input_length
        self.hid_dim = hid_dim

    def forward(self, data):
        B, T, N, C = data.shape
        
        # (B, T, N, C) -> (B, N, T*C)
        data_flat = rearrange(data, 'b t n c -> b n (t c)')
        
        outputs = []
        for i in range(self.predict_length):
            step_pred = self.predictors[i](data_flat)  # (B, N, pre_dim)
            outputs.append(step_pred.unsqueeze(1))
        
        # (B, T', N, pre_dim)
        output = torch.cat(outputs, dim=1)
        
        return output

class richer_predictor(nn.Module):
    """
    Legacy richer predictor used by the 17.85 baseline.

    这是 exp07052040 (Test MAE = 17.8548) baseline 使用的 Predictor 结构：
        predict_unit:      Linear(T*hid_dim -> d_out) + GELU   # 共享特征提取
        step_embedding:    Embedding(T', d_out)                # legacy 加性融合
        predict_unit_list: T' × Linear(d_out -> pre_dim)       # 每步独立投影

    注意：加性 step embedding 后直接接每步独立 Linear 时，其贡献可被该
    Linear 的 bias 吸收，因此不能声称它显式建模了预测步依赖。真正非冗余的
    horizon conditioning 见 horizon_conditioned_predictor。
    """
    def __init__(self, num_nodes, input_length, predict_length, hid_dim, pre_dim,
                 num_of_filters=1024, activation='gelu',
                 use_future_time_decoder=False, use_step_embedding=True):
        super(richer_predictor, self).__init__()
        self.input_length = input_length
        self.predict_length = predict_length
        self.hid_dim = hid_dim
        self.use_future_time_decoder = use_future_time_decoder
        self.use_step_embedding = use_step_embedding
        assert activation in ['gelu', 'relu']
        self.predict_unit = nn.Sequential(
            nn.Linear(input_length * hid_dim, num_of_filters),
            nn.GELU() if activation == 'gelu' else nn.ReLU()
        )
        if self.use_step_embedding:
            self.step_embedding = nn.Embedding(predict_length, num_of_filters)
        self.predict_unit_list = nn.ModuleList(
            [nn.Linear(num_of_filters, pre_dim) for _ in range(predict_length)]
        )
        if self.use_future_time_decoder:
            # run.py performs a second, historical global Xavier pass after the
            # model is built. Preserve the RNG state around this extra module so
            # that pass starts from exactly the same state as the baseline.
            rng_state = torch.random.get_rng_state()
            self.future_time_modulation = nn.Linear(hid_dim, 2 * num_of_filters)
            nn.init.zeros_(self.future_time_modulation.weight)
            nn.init.zeros_(self.future_time_modulation.bias)
            torch.random.set_rng_state(rng_state)
            # run.py applies the historical global Xavier initialization. This
            # branch must stay at zero so enabling it initially reproduces the
            # baseline output exactly.
            self.future_time_modulation.preserve_initialization = True

    def forward(self, data, future_time_emb=None):
        B, T, N, C = data.shape
        if T != self.input_length or C != self.hid_dim:
            raise ValueError(
                f'richer_predictor expected data shape (*, {self.input_length}, *, '
                f'{self.hid_dim}), got {tuple(data.shape)}'
            )
        if self.use_future_time_decoder:
            if future_time_emb is None:
                raise ValueError(
                    'future_time_emb is required when use_future_time_decoder=True'
                )
            if (future_time_emb.dim() != 4
                    or future_time_emb.shape[0] != B
                    or future_time_emb.shape[1] != self.predict_length
                    or future_time_emb.shape[2] not in (1, N)
                    or future_time_emb.shape[3] != self.hid_dim):
                raise ValueError(
                    'future_time_emb must have shape '
                    f'(B, {self.predict_length}, 1|N, {self.hid_dim}), '
                    f'got {tuple(future_time_emb.shape)}'
                )

        # (B, T, N, C) -> (B, N, T*C)
        data_flat = rearrange(data, 'b t n c -> b n (t c)')
        h = self.predict_unit(data_flat)  # (B, N, num_of_filters)
        outputs = []
        for j in range(self.predict_length):
            if self.use_step_embedding:
                step_emb = self.step_embedding.weight[j].view(1, 1, -1)
                h_j = h + step_emb  # legacy additive fusion
            else:
                h_j = h
            if self.use_future_time_decoder:
                # future_time_emb[:, j]: (B, 1|N, hid_dim), broadcast over N if needed.
                gamma, beta = self.future_time_modulation(
                    future_time_emb[:, j]
                ).chunk(2, dim=-1)
                h_j = h_j * (1.0 + torch.tanh(gamma)) + beta
            out_j = self.predict_unit_list[j](h_j)  # (B, N, pre_dim)
            outputs.append(out_j.unsqueeze(1))  # (B, 1, N, pre_dim)
        output = torch.cat(outputs, dim=1)  # (B, T', N, pre_dim)
        return output


class SignedIncrementHorizonMixer(nn.Module):
    """Locally couple signed forecast increments across nearby horizons.

    The predictor supplies cumulative changes relative to the last observed
    value.  This module converts them to signed step increments, applies a
    bounded banded horizon transform, and integrates them back to a cumulative
    trajectory.  Its zero parameterization is exactly the identity, which
    permits fair initialization from an existing checkpoint.
    """

    def __init__(self, predict_length, radius=2, max_scale=0.25):
        super().__init__()
        self.predict_length = int(predict_length)
        self.radius = int(radius)
        self.max_scale = float(max_scale)
        if self.predict_length < 1:
            raise ValueError('predict_length must be positive')
        if not 0 <= self.radius < self.predict_length:
            raise ValueError(
                'signed-increment mixer radius must be in '
                f'[0, {self.predict_length - 1}], got {self.radius}'
            )
        if self.max_scale <= 0.0:
            raise ValueError('signed-increment mixer max_scale must be positive')

        self.raw_mixing = nn.Parameter(torch.zeros(
            self.predict_length, self.predict_length
        ))
        horizon = torch.arange(self.predict_length)
        band_mask = (
            torch.abs(horizon[:, None] - horizon[None, :]) <= self.radius
        ).to(torch.float32)
        self.register_buffer(
            'identity', torch.eye(self.predict_length), persistent=True
        )
        self.register_buffer('band_mask', band_mask, persistent=True)
        # run.py must not replace the exact zero/identity initialization.
        self.preserve_initialization = True

    def forward(self, cumulative_change):
        if cumulative_change.ndim != 4:
            raise ValueError(
                'cumulative_change must have shape (B, T, N, C)'
            )
        if cumulative_change.shape[1] != self.predict_length:
            raise ValueError(
                'signed-increment mixer horizon mismatch: '
                f'{cumulative_change.shape[1]} != {self.predict_length}'
            )
        signed_increment = torch.cat([
            cumulative_change[:, :1],
            cumulative_change[:, 1:] - cumulative_change[:, :-1],
        ], dim=1)
        mixing = self.identity + (
            self.max_scale * torch.tanh(self.raw_mixing) * self.band_mask
        )
        mixed_increment = torch.einsum(
            'st,btnc->bsnc', mixing, signed_increment
        )
        return torch.cumsum(mixed_increment, dim=1)


class ElasticRankRouter(nn.Module):
    """Tiny observed-history router for a nested low/full-rank predictor."""

    def __init__(self, hidden=8, initial_high_probability=0.8):
        super().__init__()
        probability = float(initial_high_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError('initial_high_probability must be in (0, 1)')
        self.network = nn.Sequential(
            nn.Linear(6, int(hidden)), nn.GELU(), nn.Linear(int(hidden), 1)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, float(np.log(
            probability / (1.0 - probability)
        )))
        self.preserve_initialization = True

    def forward(self, history, temperature=1.0):
        features = self.features(history)
        logits = self.network(features).squeeze(-1)
        return torch.sigmoid(logits / max(float(temperature), 1e-4))

    @staticmethod
    def features(history):
        values = history.float()
        scale = values.abs().mean(dim=(1, 2, 3)).clamp_min(1e-4)
        mean = values.mean(dim=(1, 2, 3)) / scale
        std = values.std(dim=(1, 2, 3), unbiased=False) / scale
        difference = values[:, 1:] - values[:, :-1]
        mean_abs_diff = difference.abs().mean(dim=(1, 2, 3)) / scale
        last_abs_diff = difference[:, -1].abs().mean(dim=(1, 2)) / scale
        trend = (values[:, -1] - values[:, 0]).abs().mean(dim=(1, 2)) / scale
        zero_fraction = values.eq(0).float().mean(dim=(1, 2, 3))
        features = torch.stack([
            mean, std, mean_abs_diff, last_abs_diff, trend, zero_fraction,
        ], dim=-1)
        return features


class factorized_richer_predictor(nn.Module):
    """Low-rank deployment replacement for :class:`richer_predictor`.

    The expensive ``Linear(T * hid_dim, d_out)`` projection is replaced by
    two linear maps with no non-linearity between them::

        Linear(T * hid_dim, factor_rank, bias=False)
        Linear(factor_rank, d_out, bias=True)
        GELU

    This topology can be initialized from an already-trained richer predictor
    with a truncated SVD. The legacy step embedding is not stored: because
    each horizon has its own linear readout, ``W_j (h + e_j) + b_j`` is exactly
    equivalent to ``W_j h + (b_j + W_j e_j)``. Folding it into the horizon
    bias removes parameters without changing the teacher function beyond the
    deliberately truncated shared projection.
    """

    def __init__(self, num_nodes, input_length, predict_length, hid_dim,
                 pre_dim, num_of_filters=1024, factor_rank=256,
                 activation='gelu', use_signed_increment_mixer=False,
                 signed_increment_mixer_radius=2,
                 signed_increment_mixer_max_scale=0.25):
        super().__init__()
        del num_nodes  # Predictor weights are shared over nodes.
        self.input_length = int(input_length)
        self.predict_length = int(predict_length)
        self.hid_dim = int(hid_dim)
        self.num_of_filters = int(num_of_filters)
        self.factor_rank = int(factor_rank)
        input_features = self.input_length * self.hid_dim
        max_rank = min(input_features, self.num_of_filters)
        if not 1 <= self.factor_rank <= max_rank:
            raise ValueError(
                f'factor_rank must be in [1, {max_rank}], got '
                f'{self.factor_rank}'
            )
        if activation not in ('gelu', 'relu'):
            raise ValueError(f'Unsupported activation: {activation}')

        self.predict_unit = nn.Sequential(
            nn.Linear(input_features, self.factor_rank, bias=False),
            nn.Linear(self.factor_rank, self.num_of_filters, bias=True),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
        )
        self.predict_unit_list = nn.ModuleList([
            nn.Linear(self.num_of_filters, pre_dim)
            for _ in range(self.predict_length)
        ])
        self.use_signed_increment_mixer = bool(use_signed_increment_mixer)
        if self.use_signed_increment_mixer:
            self.signed_increment_mixer = SignedIncrementHorizonMixer(
                predict_length=self.predict_length,
                radius=signed_increment_mixer_radius,
                max_scale=signed_increment_mixer_max_scale,
            )
        self.elastic_low_rank = None
        self.elastic_target_high_fraction = 0.5
        self.elastic_route_mode = 'batch_topk'
        self.elastic_force_rank = None
        self.last_elastic_probability = None
        self.last_elastic_hard_gate = None
        self.last_elastic_low_output = None
        self.last_elastic_high_output = None

    def configure_elastic_rank(
        self, low_rank, target_high_fraction=0.5,
        route_mode='batch_topk',
    ):
        low_rank = int(low_rank)
        if not 1 <= low_rank < self.factor_rank:
            raise ValueError(
                f'elastic low rank must be in [1, {self.factor_rank - 1}]'
            )
        self.elastic_low_rank = low_rank
        self.elastic_target_high_fraction = float(target_high_fraction)
        if route_mode not in ('batch_topk', 'independent_threshold'):
            raise ValueError('Unsupported elastic route mode')
        self.elastic_route_mode = route_mode
        self.register_buffer(
            'elastic_routing_threshold', torch.tensor(0.5)
        )
        if not 0.0 <= self.elastic_target_high_fraction <= 1.0:
            raise ValueError('target_high_fraction must be in [0, 1]')

    def _hard_budget_gate(self, probability):
        batch_size = probability.numel()
        if self.elastic_force_rank is not None:
            if self.elastic_force_rank not in (
                self.elastic_low_rank, self.factor_rank
            ):
                raise ValueError('elastic_force_rank must select a subnet')
            return torch.full_like(
                probability,
                float(self.elastic_force_rank == self.factor_rank),
            )
        if self.elastic_route_mode == 'independent_threshold':
            return probability.ge(self.elastic_routing_threshold).to(
                probability.dtype
            )
        high_count = int(round(
            batch_size * self.elastic_target_high_fraction
        ))
        high_count = max(0, min(batch_size, high_count))
        gate = torch.zeros_like(probability)
        if high_count:
            indices = torch.topk(
                probability, k=high_count, sorted=False
            ).indices
            gate[indices] = 1.0
        return gate

    def _readout(self, hidden):
        output = torch.cat([
            head(hidden).unsqueeze(1) for head in self.predict_unit_list
        ], dim=1)
        if self.use_signed_increment_mixer:
            output = self.signed_increment_mixer(output)
        return output

    def forward(self, data, elastic_probability=None):
        _, time_steps, _, channels = data.shape
        if time_steps != self.input_length or channels != self.hid_dim:
            raise ValueError(
                'factorized_richer_predictor expected data shape '
                f'(*, {self.input_length}, *, {self.hid_dim}), got '
                f'{tuple(data.shape)}'
            )
        data_flat = rearrange(data, 'b t n c -> b n (t c)')
        if elastic_probability is not None and self.elastic_low_rank is not None:
            projection_in = self.predict_unit[0]
            projection_out = self.predict_unit[1]
            activation = self.predict_unit[2]
            hard_vector = self._hard_budget_gate(elastic_probability)
            if not self.training:
                # Execute only the selected subnet for each sample. This is
                # the deployable compute path; unlike zero masking it skips
                # the omitted projection columns for rank-64 samples.
                output = data_flat.new_empty(
                    data_flat.shape[0], self.predict_length,
                    data_flat.shape[1], self.predict_unit_list[0].out_features,
                )
                low_indices = hard_vector.eq(0).nonzero(as_tuple=True)[0]
                high_indices = hard_vector.eq(1).nonzero(as_tuple=True)[0]
                if low_indices.numel():
                    low_latent = F.linear(
                        data_flat[low_indices],
                        projection_in.weight[:self.elastic_low_rank],
                    )
                    low_hidden = activation(F.linear(
                        low_latent,
                        projection_out.weight[:, :self.elastic_low_rank],
                        projection_out.bias,
                    ))
                    output[low_indices] = self._readout(low_hidden)
                if high_indices.numel():
                    high_hidden = activation(projection_out(
                        projection_in(data_flat[high_indices])
                    ))
                    output[high_indices] = self._readout(high_hidden)
                self.last_elastic_probability = elastic_probability
                self.last_elastic_hard_gate = hard_vector
                self.last_elastic_low_output = None
                self.last_elastic_high_output = None
                return output
            latent = projection_in(data_flat)
            high_hidden = activation(projection_out(latent))
            low_hidden = activation(F.linear(
                latent[..., :self.elastic_low_rank],
                projection_out.weight[:, :self.elastic_low_rank],
                projection_out.bias,
            ))
            low_output = self._readout(low_hidden)
            high_output = self._readout(high_hidden)
            probability = elastic_probability.reshape(-1, 1, 1, 1).to(
                dtype=high_output.dtype
            )
            hard_gate = hard_vector.reshape(-1, 1, 1, 1).to(high_output.dtype)
            gate = hard_gate
            if self.training:
                gate = hard_gate + probability - probability.detach()
            output = low_output + gate * (high_output - low_output)
            hidden_gate = gate[:, 0]
            self.last_shared_hidden = (
                low_hidden + hidden_gate * (high_hidden - low_hidden)
            )
            self.last_elastic_probability = elastic_probability
            self.last_elastic_hard_gate = hard_vector
            self.last_elastic_low_output = low_output
            self.last_elastic_high_output = high_output
            return output

        hidden = self.predict_unit(data_flat)
        # Expose the shared predictor representation for optional training-only
        # horizon-gradient diagnostics/balancing.  It is overwritten on every
        # forward pass and is never used by inference.
        self.last_shared_hidden = hidden
        self.last_elastic_probability = None
        self.last_elastic_hard_gate = None
        self.last_elastic_low_output = None
        self.last_elastic_high_output = None
        return self._readout(hidden)

    @torch.no_grad()
    def initialize_from_richer_state_dict(
        self,
        state_dict,
        prefix='',
        activation_second_moment=None,
        activation_ridge=1e-5,
    ):
        """Initialize from a legacy richer-predictor state dict.

        Returns the fraction of the original projection's squared singular-
        value energy retained by the selected rank.
        """
        projection_weight_key = prefix + 'predict_unit.0.weight'
        projection_bias_key = prefix + 'predict_unit.0.bias'
        required = {projection_weight_key, projection_bias_key}
        for horizon in range(self.predict_length):
            required.add(prefix + f'predict_unit_list.{horizon}.weight')
            required.add(prefix + f'predict_unit_list.{horizon}.bias')
        missing = sorted(required.difference(state_dict))
        if missing:
            raise KeyError(
                'Teacher checkpoint is missing richer-predictor keys: '
                f'{missing}'
            )

        source_weight = state_dict[projection_weight_key].detach().to(
            device='cpu', dtype=torch.float32
        )
        source_bias = state_dict[projection_bias_key].detach().to(
            device='cpu', dtype=torch.float32
        )
        expected_weight_shape = (
            self.num_of_filters, self.input_length * self.hid_dim
        )
        if tuple(source_weight.shape) != expected_weight_shape:
            raise ValueError(
                f'Unexpected teacher projection shape '
                f'{tuple(source_weight.shape)}; expected '
                f'{expected_weight_shape}'
            )
        if tuple(source_bias.shape) != (self.num_of_filters,):
            raise ValueError(
                f'Unexpected teacher projection bias shape '
                f'{tuple(source_bias.shape)}'
            )

        if activation_second_moment is None:
            matrix_to_factor = source_weight
            right_unweight = None
        else:
            # Minimize E[||(W - BA)x||^2] instead of ||W - BA||_F^2.
            # For C=E[xx^T]=LL^T, truncated SVD is applied to W L and the
            # right factor is subsequently multiplied by L^{-1}.
            covariance = torch.as_tensor(
                activation_second_moment,
                device='cpu', dtype=torch.float64,
            )
            expected_covariance_shape = (
                expected_weight_shape[1], expected_weight_shape[1]
            )
            if tuple(covariance.shape) != expected_covariance_shape:
                raise ValueError(
                    f'Unexpected activation second-moment shape '
                    f'{tuple(covariance.shape)}; expected '
                    f'{expected_covariance_shape}'
                )
            if not torch.isfinite(covariance).all():
                raise ValueError('Activation second moment contains NaN/Inf')
            covariance = 0.5 * (covariance + covariance.T)
            diagonal_scale = covariance.diagonal().mean().clamp_min(
                torch.finfo(covariance.dtype).eps
            )
            ridge = float(activation_ridge)
            if ridge <= 0.0:
                raise ValueError('activation_ridge must be positive')
            covariance = covariance + (
                ridge * diagonal_scale
                * torch.eye(covariance.shape[0], dtype=covariance.dtype)
            )
            right_unweight = torch.linalg.cholesky(covariance)
            matrix_to_factor = (
                source_weight.to(torch.float64) @ right_unweight
            )

        # Symmetric square-root split keeps the two factors similarly scaled.
        u, singular_values, vh = torch.linalg.svd(
            matrix_to_factor, full_matrices=False
        )
        rank = self.factor_rank
        root_s = singular_values[:rank].sqrt()
        factor_in = root_s[:, None] * vh[:rank]
        if right_unweight is not None:
            # factor_in @ L^{-1}; solve L^T X = factor_in^T instead of
            # explicitly forming an inverse.
            factor_in = torch.linalg.solve_triangular(
                right_unweight.T,
                factor_in.T,
                upper=True,
            ).T
        factor_out = u[:, :rank] * root_s[None, :]
        self.predict_unit[0].weight.copy_(factor_in.to(
            device=self.predict_unit[0].weight.device,
            dtype=self.predict_unit[0].weight.dtype,
        ))
        self.predict_unit[1].weight.copy_(factor_out.to(
            device=self.predict_unit[1].weight.device,
            dtype=self.predict_unit[1].weight.dtype,
        ))
        self.predict_unit[1].bias.copy_(source_bias.to(
            device=self.predict_unit[1].bias.device,
            dtype=self.predict_unit[1].bias.dtype,
        ))

        step_key = prefix + 'step_embedding.weight'
        if step_key in state_dict:
            step_embedding = state_dict[step_key].detach().to(
                device='cpu', dtype=torch.float32
            )
            expected_step_shape = (
                self.predict_length, self.num_of_filters
            )
            if tuple(step_embedding.shape) != expected_step_shape:
                raise ValueError(
                    f'Unexpected teacher step embedding shape '
                    f'{tuple(step_embedding.shape)}; expected '
                    f'{expected_step_shape}'
                )
        else:
            step_embedding = torch.zeros(
                self.predict_length, self.num_of_filters,
                dtype=torch.float32,
            )

        for horizon, target_head in enumerate(self.predict_unit_list):
            weight = state_dict[
                prefix + f'predict_unit_list.{horizon}.weight'
            ].detach().to(device='cpu', dtype=torch.float32)
            bias = state_dict[
                prefix + f'predict_unit_list.{horizon}.bias'
            ].detach().to(device='cpu', dtype=torch.float32)
            if tuple(weight.shape) != tuple(target_head.weight.shape):
                raise ValueError(
                    f'Unexpected teacher horizon-{horizon} weight shape '
                    f'{tuple(weight.shape)}; expected '
                    f'{tuple(target_head.weight.shape)}'
                )
            if tuple(bias.shape) != tuple(target_head.bias.shape):
                raise ValueError(
                    f'Unexpected teacher horizon-{horizon} bias shape '
                    f'{tuple(bias.shape)}; expected '
                    f'{tuple(target_head.bias.shape)}'
                )
            folded_bias = bias + weight.matmul(step_embedding[horizon])
            target_head.weight.copy_(weight.to(
                device=target_head.weight.device,
                dtype=target_head.weight.dtype,
            ))
            target_head.bias.copy_(folded_bias.to(
                device=target_head.bias.device,
                dtype=target_head.bias.dtype,
            ))

        total_energy = singular_values.square().sum().clamp_min(
            torch.finfo(singular_values.dtype).eps
        )
        retained_energy = singular_values[:rank].square().sum() / total_energy
        return float(retained_energy.item())


class dlinear_decoder(nn.Module):
    """DLinear-style trend/seasonal decoupled readout.

    Replaces richer_predictor's per-horizon independent heads.  Motivation:
    12 independent Linear heads each regress their own horizon from a shared
    feature, so no head sees the cross-horizon structure and late horizons
    drift toward the mean.  Here every (node, feature-channel) pair is treated
    as an independent DLinear channel: a moving-average trend branch and a
    residual seasonal branch, each mapped by ONE shared bias-free
    Linear(input_length -> predict_length) on the time axis.  A single shared
    Linear(hid_dim -> pre_dim) readout is applied to every horizon.  All
    horizons therefore share both the time mixing and the output projection
    (cross-horizon coupling), which richer_predictor lacks.

    This follows DLinear (Zeng et al., AAAI 2023) exactly for the
    decomposition: edge-replicated padding before AvgPool1d, bias-free
    per-branch linears, additive recombination.
    """

    def __init__(self, num_nodes, input_length, predict_length, hid_dim, pre_dim,
                 moving_avg_kernel=3):
        super().__init__()
        self.input_length = input_length
        self.predict_length = predict_length
        self.hid_dim = hid_dim
        self.pre_dim = pre_dim
        kernel = int(moving_avg_kernel)
        if kernel < 1 or kernel % 2 == 0 or kernel > input_length:
            raise ValueError(
                f'moving_avg_kernel must be odd and in [1, {input_length}], '
                f'got {kernel}'
            )
        self.moving_avg_kernel = kernel
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel, stride=1)
        # DLinear convention: bias-free per-branch time-axis mapping.
        self.trend_linear = nn.Linear(input_length, predict_length, bias=False)
        self.seasonal_linear = nn.Linear(input_length, predict_length, bias=False)
        self.readout = nn.Linear(hid_dim, pre_dim)

    def _moving_avg(self, x):
        # x: (B, NC, T).  Edge-replicated padding, matching DLinear's
        # moving_avg so boundary averages are not diluted by zeros.
        k = self.moving_avg_kernel
        front = x[:, :, :1].repeat(1, 1, (k - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (k - 1) // 2)
        return self.avg_pool(torch.cat([front, x, end], dim=2))

    def forward(self, data, future_time_emb=None):
        B, T, N, C = data.shape
        if T != self.input_length or C != self.hid_dim:
            raise ValueError(
                f'dlinear_decoder expected data shape (*, {self.input_length}, '
                f'*, {self.hid_dim}), got {tuple(data.shape)}'
            )
        # (B, T, N, C) -> (B, N*C, T): channel-independent time axis mixing.
        x = rearrange(data, 'b t n c -> b (n c) t')
        trend = self._moving_avg(x)
        seasonal = x - trend
        out = self.trend_linear(trend) + self.seasonal_linear(seasonal)
        # (B, N*C, T') -> (B, T', N, C) -> shared readout per horizon.
        out = rearrange(out, 'b (n c) t -> b t n c', n=N, c=C)
        return self.readout(out)  # (B, T', N, pre_dim)


class horizon_conditioned_predictor(nn.Module):
    """Shared history encoder with non-redundant horizon conditioning.

    An additive step embedding followed immediately by an independent linear
    head can be absorbed into that head's bias. Applying a non-linearity after
    adding the horizon embedding makes the conditioning affect feature
    selection and therefore changes the representable function.
    """

    def __init__(self, num_nodes, input_length, predict_length, hid_dim, pre_dim,
                 num_of_filters=1024, activation='gelu'):
        super().__init__()
        self.input_length = input_length
        self.predict_length = predict_length
        self.hid_dim = hid_dim
        assert activation in ['gelu', 'relu']
        self.activation = F.gelu if activation == 'gelu' else F.relu
        self.predict_unit = nn.Sequential(
            nn.Linear(input_length * hid_dim, num_of_filters),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
        )
        self.step_embedding = nn.Embedding(predict_length, num_of_filters)
        self.predict_unit_list = nn.ModuleList(
            [nn.Linear(num_of_filters, pre_dim) for _ in range(predict_length)]
        )

    def forward(self, data):
        B, T, N, C = data.shape
        if T != self.input_length or C != self.hid_dim:
            raise ValueError(
                f'horizon_conditioned_predictor expected (*, {self.input_length}, '
                f'*, {self.hid_dim}), got {tuple(data.shape)}'
            )
        data_flat = rearrange(data, 'b t n c -> b n (t c)')
        history = self.predict_unit(data_flat)
        outputs = []
        for horizon in range(self.predict_length):
            step = self.step_embedding.weight[horizon].view(1, 1, -1)
            conditioned = self.activation(history + step)
            outputs.append(
                self.predict_unit_list[horizon](conditioned).unsqueeze(1)
            )
        return torch.cat(outputs, dim=1)

class cross_time_mlp(nn.Module):
    """
    跨时间通道混合 MLP 预测头（richer_predictor 的加深+强制跨步交互版）

    richer_predictor 在共享单层 MLP 之后，12 个预测步各自走一个独立 Linear 读出，
    步间不再交互。本头把展平后的 (B, N, 12*C) 过 **2 层 GELU MLP**，并让最终 head
    一次性输出全部 12 步（共享权重），从而强制 12 个预测步在表征层反复混合，
    而非独立读出。步序由展平位置自然编码，无需额外 step_embedding。
    输入: (B, T, N, C)   输出: (B, T, N, pre_dim)
    """
    def __init__(self, num_nodes, input_length, predict_length, hid_dim, pre_dim,
                 num_of_filters=1024, activation='gelu'):
        super(cross_time_mlp, self).__init__()
        self.predict_length = predict_length
        self.pre_dim = pre_dim
        self.hid_dim = hid_dim
        assert activation in ['gelu', 'relu']
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        # 展平 (B, T, N, C) -> (B, N, T*C) 后过 2 层 MLP，强制跨时间+跨通道混合
        self.mlp = nn.Sequential(
            nn.Linear(predict_length * hid_dim, num_of_filters),
            act,
            nn.Linear(num_of_filters, num_of_filters),
            act,
        )
        # 共享 head：一次性输出全部 12 步（T*pre_dim），步序由展平位置自然编码
        self.head = nn.Linear(num_of_filters, predict_length * pre_dim)

    def forward(self, data):
        B, T, N, C = data.shape
        # (B, T, N, C) -> (B, N, T*C)：跨时间维展平，使 MLP 能混合各预测步
        data_flat = rearrange(data, 'b t n c -> b n (t c)')
        h = self.mlp(data_flat)                 # (B, N, num_of_filters)
        out_flat = self.head(h)                 # (B, N, T*pre_dim)
        output = rearrange(out_flat, 'b n (t p) -> b t n p', t=T, p=self.pre_dim)
        return output

class ar_predictor(nn.Module):
    """
    自回归(Autoregressive)残差预测头 —— 顺序感知，攻击多 horizon 漂移。

    设计动机：richer_predictor 的 12 个预测步各自走独立 Linear 读出，步间无方向性依赖，
    导致远端 step(如 t+6/t+12) 易出现漂移 / 回归均值。本头在 richer 的无条件基预测之上，
    增加一个**顺序感知的残差修正**：第 k 步的修正量显式依赖第 k-1 步的预测值，
    从而建模 "t+1 影响 t+6" 的方向性时序依赖。

    关键点：
      - 残差式：refined_k = base_k + corr_k，基预测仍是 richer，稳定不爆炸；
      - 顺序感知：corr_k = ar_mlp(concat(base_k, refined_{k-1}))，严格按时间顺序，
        与已证伪的 cross_time_mlp(对称 flat 混合、丢顺序) 本质不同；
      - 不使用 Transformer / Conv1D，遵守轻量原则；
      - 训练与推理一致采用 free-running(用上一步 refined 输出)，无 train/test 不匹配。
    输入: (B, T, N, C)   输出: (B, T, N, pre_dim)
    """
    def __init__(self, num_nodes, input_length, predict_length, hid_dim, pre_dim,
                 num_of_filters=1024, activation='gelu'):
        super(ar_predictor, self).__init__()
        self.predict_length = predict_length
        self.pre_dim = pre_dim
        # ---- base: 与 richer_predictor 同构的无条件基预测 ----
        assert activation in ['gelu', 'relu']
        self.predict_unit = nn.Sequential(
            nn.Linear(input_length * hid_dim, num_of_filters),
            nn.GELU() if activation == 'gelu' else nn.ReLU()
        )
        self.step_embedding = nn.Embedding(predict_length, num_of_filters)
        self.predict_unit_list = nn.ModuleList(
            [nn.Linear(num_of_filters, pre_dim) for _ in range(predict_length)]
        )
        # ---- AR 残差修正：用上一步预测值修正当前步 ----
        self.ar_resid = nn.Sequential(
            nn.Linear(2 * pre_dim, pre_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(pre_dim, pre_dim)
        )

    def forward(self, data):
        B, T, N, C = data.shape
        # base 无条件预测（与 richer 一致）
        data_flat = rearrange(data, 'b t n c -> b n (t c)')
        h = self.predict_unit(data_flat)  # (B, N, num_of_filters)
        base = []
        for j in range(self.predict_length):
            step_emb = self.step_embedding(
                torch.arange(j, j + 1, device=data.device)
            )
            h_j = h + step_emb
            out_j = self.predict_unit_list[j](h_j)  # (B, N, pre_dim)
            base.append(out_j)
        base = torch.stack(base, dim=1)  # (B, T, N, pre_dim)

        # AR 顺序感知残差修正（free-running：用上一步 refined 输出，训练/推理一致）
        out = [base[:, 0]]  # 第一步无前置，直接用 base
        for j in range(1, self.predict_length):
            ctx = torch.cat([base[:, j], out[j - 1]], dim=-1)  # (B, N, 2*pre_dim)
            corr = self.ar_resid(ctx)
            out.append(base[:, j] + corr)
        output = torch.stack(out, dim=1)  # (B, T, N, pre_dim)
        return output

class getTimeLagEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, emb_dropout = 0.1):
        super().__init__()
        self.time_lag_mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, time_emb, ref_emb):
        time_lag_emb = self.time_lag_mlp(time_emb - ref_emb)
        time_lag_emb = self.dropout(time_lag_emb)
        return time_lag_emb


class getTimeEmbedding(nn.Module):
    def __init__(self, hid_dim, slice_size_per_day,emb_dropout=0.1):
        super().__init__()
        self.time_in_day_embedding = nn.Embedding(slice_size_per_day, hid_dim)
        self.day_in_week_embedding = nn.Embedding(7, hid_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, time_data):
        t_hour = time_data[...,0:1]
        t_day = time_data[...,1:2]
        time_in_day_emb = self.time_in_day_embedding(t_hour)
        day_in_week_emb = self.day_in_week_embedding(t_day)
        time_emb = time_in_day_emb + day_in_week_emb
        time_emb = self.dropout(time_emb)    
        return time_emb
        
class getSpatialEmbedding(nn.Module):
    def __init__(self, hid_dim, num_nodes, emb_dropout=0.1):
        super().__init__()
        self.spatial_embedding = nn.Embedding(num_nodes, hid_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x, spatial_indexs=None):
        if spatial_indexs is None:
            batch, _,  num_nodes, _ = x.shape
            spatial_indexs = torch.LongTensor(torch.arange(num_nodes)).to(x.device)  # (N,)
        spatial_emb = self.spatial_embedding(spatial_indexs).unsqueeze(0).unsqueeze(1)  # (N, d)->(1, 1, N, d)
        spatial_emb = self.dropout(spatial_emb)
        return spatial_emb


class DataEncoding(nn.Module):
    def __init__(self, in_dim, hid_dim, hasCross=True, activation='relu'):
        super().__init__()
        assert activation in ['gelu', 'relu']
        in_units = in_dim*2 if hasCross else in_dim
        self.hasCross = hasCross
        self.linear1 = nn.Linear(in_units, hid_dim)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x, latestX):
        if self.hasCross:
            data = torch.cat([x, latestX], dim=-1)
        else:
            data = x
        data = self.linear1(data)
        data = self.activation(data)
        data = self.linear2(data)
        return data


class SignedChangeEncoding(nn.Module):
    """Encode observed first differences as a gated latent input stream.

    The gate is initialized at zero, so enabling this module reproduces the
    parent model exactly before fine-tuning. Only observed history is used;
    there is no future-label gate or post-hoc correction at inference.
    """

    def __init__(self, in_dim, hid_dim, activation='gelu', max_scale=0.5):
        super().__init__()
        if activation not in ('gelu', 'relu'):
            raise ValueError(f'Unsupported activation: {activation}')
        self.max_scale = float(max_scale)
        if self.max_scale <= 0.0:
            raise ValueError('signed-change encoding max_scale must be positive')
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        self.channel_gate = nn.Parameter(torch.zeros(hid_dim))
        # Preserve the exact zero gate and this branch's local initialization.
        self.preserve_initialization = True

    def forward(self, history):
        if history.ndim != 4:
            raise ValueError('history must have shape (B, T, N, C)')
        first_difference = torch.cat([
            torch.zeros_like(history[:, :1]),
            history[:, 1:] - history[:, :-1],
        ], dim=1)
        encoded_change = self.encoder(first_difference)
        gate = self.max_scale * torch.tanh(self.channel_gate)
        return encoded_change * gate.view(1, 1, 1, -1)


class ShortWindowDynamicsEncoding(nn.Module):
    """Encode six statistics from the observed 12-step window."""

    def __init__(self, hid_dim, max_scale=0.5):
        super().__init__()
        self.max_scale = float(max_scale)
        self.encoder = nn.Sequential(
            nn.Linear(6, hid_dim), nn.GELU(), nn.Linear(hid_dim, hid_dim),
        )
        self.channel_gate = nn.Parameter(torch.zeros(hid_dim))
        self.preserve_initialization = True

    def forward(self, history):
        if history.ndim != 4 or history.shape[1] != 12 or history.shape[-1] != 1:
            raise ValueError('short-window dynamics requires (B, 12, N, 1)')
        x = history[..., 0]
        d = x[:, 1:] - x[:, :-1]
        stats = torch.stack((
            x.amax(dim=1) - x.amin(dim=1),
            x.std(dim=1, unbiased=False),
            d.abs().mean(dim=1),
            d.abs().amax(dim=1),
            x[:, -1] - x[:, 0],
            x[:, 6:].mean(dim=1) - x[:, :6].mean(dim=1),
        ), dim=-1)
        gate = self.max_scale * torch.tanh(self.channel_gate)
        encoded = self.encoder(stats) * gate.view(1, 1, -1)
        return encoded[:, None].expand(-1, history.shape[1], -1, -1)


# ========================= 新增类定义（第一部分）=========================
# 包含：MS_TCN（多尺度时间卷积）、FTLE（全时滞嵌入）、MAPA_EncoderLayer（记忆增强代理注意力）
# ================================================================================


class OriginalTCN(nn.Module):
    """Single-scale temporal convolution used by models/TLAST.py."""

    def __init__(self, hid_dim, tau):
        super().__init__()
        if tau not in (2, 3):
            raise ValueError(f'Original TLAST TCN supports tau=2/3, got {tau}')
        self.padding = nn.ReplicationPad2d((tau - 2, 1, 0, 0))
        self.conv = nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, tau))

    def forward(self, x):
        return self.conv(self.padding(x))


class MS_TCN(nn.Module):
    """
    多尺度时间卷积模块（Multi-Scale Temporal Convolution Network）
    
    功能：
        使用多个并行的1D卷积分支（不同tau值）捕获多尺度时间模式，
        然后通过1×1卷积融合多分支特征，替代原始TLAST中单一tau的TCN。
    
    输入形状：
        x: (B, C, N, T)  # B=batch, C=hid_dim, N=num_nodes, T=input_length
    输出形状：
        out: (B, C, N, T)  # 与输入形状相同（时间维度T保持不变）
    """
    def __init__(self, hid_dim, tau_list=[3, 5, 7], activation='relu', dilation_list=None,
                 num_nodes=None, node_mix=False):
        """
        参数：
            hid_dim: 隐藏维度（通道数C）
            tau_list: 多分支卷积核大小列表，默认[3,5,7]对应短/中/长期时间模式
            activation: 激活函数类型（'relu'或'gelu'）
            dilation_list: 若提供，则进入「扩张卷积模式」——每个分支固定 Conv2d(kernel=(1,3))，
                仅通过 dilation 扩大感受野，padding 按 (3-1)*dilation 计算保持输出 T 不变。
                例：dilation_list=[1,2,4] → 感受野 3/5/9，覆盖更长程时间依赖，接口与 shape 不变。
        """
        super().__init__()
        self.tau_list = tau_list
        self.hid_dim = hid_dim
        self.dilation_list = dilation_list

        # ===== 多分支并行卷积 =====
        self.branches = nn.ModuleList()
        if dilation_list is not None:
            # ----- 扩张卷积模式（最小侵入改动）-----
            # 保持 Conv2d(kernel=(1,3))，仅 dilation 变化；padding 保证 same 输出 T 不变。
            for d in dilation_list:
                pad_total = (3 - 1) * d   # = 2*d
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                branch = nn.Sequential(
                    nn.ReplicationPad2d((pad_left, pad_right, 0, 0)),
                    nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 3), padding=0,
                              dilation=(1, d)),
                    nn.ReLU() if activation == 'relu' else nn.GELU()
                )
                self.branches.append(branch)
        else:
            # ----- 原始多分支并行卷积（固定核，无扩张）-----
            # 每个分支使用不同的tau（卷积核大小），捕获不同时间尺度的模式
            for tau in tau_list:
                # 计算padding以保持时间维度T不变（对任意tau奇偶都精确等长）
                # 使用边缘复制填充（ReplicationPad2d），左右可不对称，实现"same"卷积
                # pad_left + pad_right = tau - 1  =>  Conv2d(kernel=tau) 输出长度恒为 T
                pad_total = tau - 1
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left

                branch = nn.Sequential(
                    # 边缘复制填充：仅在时间维度（左右）进行复制填充
                    # padding格式：(left, right, top, bottom) = (pad_left, pad_right, 0, 0)
                    nn.ReplicationPad2d((pad_left, pad_right, 0, 0)),
                    # Conv2d核大小(1, tau)：1表示不在节点维度做卷积，tau表示在时间维度做卷积
                    # padding=0 因为已经通过ReplicationPad2d手动填充
                    nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, tau), padding=0),
                    nn.ReLU() if activation == 'relu' else nn.GELU()
                )
                self.branches.append(branch)
        
        # ===== 特征融合层 =====
        # 将多分支输出沿通道维度拼接（C*num_branches），然后用1×1卷积降维回C
        # 注意：必须用实际分支数 len(self.branches)，而非 len(tau_list)——
        # 扩张模式下 tau_list 仍是默认值，二者长度可能不同会导致通道数对不上。
        self.fuse_conv = nn.Conv2d(hid_dim * len(self.branches), hid_dim, kernel_size=1)

        # ===== 节点混合开关（打破节点独立性，单变量 ablation）=====
        # 注意：必须用 nn.Linear(N, N) 作用在 N 维（先 transpose 到 (B,C,T,N) 再应用），
        # 而非 nn.Conv2d(kernel=(1,1)) —— 后者混合的是通道 C，不是节点 N。
        self.node_mix_flag = node_mix
        if node_mix:
            assert num_nodes is not None, "node_mix=True 必须提供 num_nodes"
            self.node_mix = nn.Linear(num_nodes, num_nodes)
    
    def forward(self, x):
        """
        前向传播
        
        参数：
            x: (B, C, N, T)  # 输入特征
        返回：
            output: (B, C, N, T)  # 输出特征（形状与输入相同）
        """
        # 存储每个分支的输出
        branch_outputs = []
        
        # ===== 多分支并行计算 =====
        for branch in self.branches:
            out = branch(x)  # (B, C, N, T) - 每个分支输出形状相同
            if self.node_mix_flag:
                # 在节点维度做线性混合：(B,C,N,T)->(B,C,T,N)->Linear(N,N)->(B,C,N,T)
                out = out.transpose(2, 3)
                out = self.node_mix(out)
                out = out.transpose(2, 3)
            branch_outputs.append(out)
        
        # ===== 特征融合 =====
        # 沿通道维度拼接所有分支输出：(B, C*num_branches, N, T)
        fused = torch.cat(branch_outputs, dim=1)
        
        # 使用1×1卷积降维回 (B, C, N, T)
        output = self.fuse_conv(fused)
        
        return output


class ModernTCNBlock(nn.Module):
    """
    ModernTCN 式时间骨干（ICLR'24 思想的简化实现），用于替换原始 MS_TCN。

    与 MS_TCN 的关键区别：
      - MS_TCN 的卷积核为 (1, tau)，只在时间维卷积，**节点(N)之间不混合**；
      - ModernTCNBlock 额外引入「节点维混合」（对应 ModernTCN 的 variate mixing：
        在交通预测里每个节点就是一条 variate），从而打破节点独立假设，并配合大核时间
        卷积覆盖全部 12 步历史。

    输入/输出形状：x: (B, C, N, T)
    """
    def __init__(self, hid_dim, num_nodes, t_kernel=11, activation='gelu'):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()

        # 1) 时间维深度可分离卷积（per feature-channel，沿 T），大核覆盖全部历史步
        self.temporal = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, t_kernel),
                      padding=(0, t_kernel // 2), groups=hid_dim),
            act,
        )
        # 2) 节点维混合（variate mixing）：在 N 轴上做线性混合
        #    （等价 ModernTCN 的 1x1 conv over variates，交通里即节点间交互）
        self.node_mix = nn.Linear(num_nodes, num_nodes)
        # 3) 通道(特征)维 FFN（MLP-mixer 式），GLU 门控
        self.ffn = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv2d(hid_dim, hid_dim, kernel_size=1),
        )
        self.drop = nn.Dropout(0.1)

    def forward(self, x):  # x: (B, C, N, T)
        # 时间分支（残差）
        x = x + self.temporal(x)
        # 节点混合：把 N 轴换到最后一维做 Linear，再换回
        x = x.transpose(2, 3)          # (B, C, T, N)
        x = self.node_mix(x)           # 在 N 维做线性混合 -> (B, C, T, N)
        x = x.transpose(2, 3)          # (B, C, N, T)
        # 通道 FFN（残差 + dropout）
        x = x + self.drop(self.ffn(x))
        return x


class FTLE(nn.Module):
    """
    全时滞嵌入模块（Full Time-Lag Embedding）
    
    功能：
        计算所有时间步两两之间的相对时滞表示（而不仅是最后一个时间步），
        通过MLP学习时滞特征，然后对所有参考时间步求平均，得到每个时间步的全时滞嵌入。
    
    输入形状：
        time_emb: (B, T, N, hid_dim)  # 时间嵌入（已包含时间-of-day和day-of-week信息）
    输出形状：
        time_lag_emb: (B, T, N, hid_dim)  # 全时滞嵌入（形状与输入相同）
    """
    def __init__(self, hid_dim, emb_dropout=0.1):
        """
        参数：
            hid_dim: 隐藏维度
            emb_dropout: dropout概率
        """
        super().__init__()
        # MLP用于学习时滞特征
        # 输入：时滞差异向量 (hid_dim,)
        # 输出：时滞嵌入向量 (hid_dim,)
        self.time_lag_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
    
    def forward(self, time_emb):
        """
        前向传播：计算全时滞嵌入
        
        参数：
            time_emb: (B, T, N, hid_dim)  # 时间嵌入
        返回：
            time_lag_emb: (B, T, N, hid_dim)  # 全时滞嵌入
        """
        B, T, N, C = time_emb.shape
        
        # ===== 计算两两时间步之间的差异 =====
        # 使用广播机制计算所有时间步对 (i, j) 的差异
        # time_emb_exp1: (B, T, 1, N, C) - 第i个时间步
        # time_emb_exp2: (B, 1, T, N, C) - 第j个时间步
        time_emb_exp1 = time_emb.unsqueeze(2)
        time_emb_exp2 = time_emb.unsqueeze(1)
        
        # 计算差异矩阵：(B, T, T, N, C)
        # time_lag[i, j] = time_emb[i] - time_emb[j] 表示时间步i相对于时间步j的时滞
        time_lag = time_emb_exp1 - time_emb_exp2
        
        # ===== 应用MLP学习时滞特征 =====
        # 将5D张量展平为2D以应用MLP：(B*T*T*N, C)
        orig_shape = time_lag.shape
        time_lag_flat = time_lag.view(-1, C)
        
        # 应用MLP：(B*T*T*N, C) -> (B*T*T*N, C)
        time_lag_emb_flat = self.time_lag_mlp(time_lag_flat)
        
        # 恢复原始形状：(B, T, T, N, C)
        time_lag_emb = time_lag_emb_flat.view(orig_shape)
        
        # ===== 聚合：对所有参考时间步求平均 =====
        # 对于每个时间步i，计算所有参考时间步j的时滞嵌入的平均值
        # 输出：(B, T, N, C)
        time_lag_emb = time_lag_emb.mean(dim=2)
        
        # ===== Dropout =====
        time_lag_emb = self.dropout(time_lag_emb)
        
        return time_lag_emb


class MAPA_EncoderLayer(nn.Module):
    """
    记忆增强原型注意力编码层（Memory-Augmented Prototype Attention Encoder Layer）
    
    功能：
        1. 使用可学习的Memory Bank（记忆库）生成代理节点（proxy），
           替代原始的基于readout_fc的生成方式。
        2. Memory Bank包含多个可学习的原型向量，通过注意力机制与节点特征交互。
        3. 计算正交损失（orthogonal loss）以防止原型之间坍缩（collapse）。
    
    输入形状：
        data: (B, T, N, hid_dim)  # 输入特征
    输出形状：
        final_out: (B, T, N, hid_dim)  # 输出特征（形状与输入相同）
        ortho_loss: scalar tensor  # 正交损失（用于训练时的正则化）
    """
    def __init__(self, factor, hid_dim, n_heads, num_nodes=None,
                 d_ff=None, dropout=0.1, att_dropout=0.1, activation='gelu',
                   query_cross_time=True, norm_flag=1, return_att=False, num_prototypes=16,
                   bypass_alpha=0.0):
        """
        参数：
            factor: 代理节点数量（保留参数以兼容原始接口，实际由num_prototypes替代）
            hid_dim: 隐藏维度
            n_heads: 注意力头数
            num_nodes: 节点数量（保留参数以兼容原始接口）
            d_ff: 前馈网络隐藏维度（默认4*hid_dim）
            dropout: dropout概率
            att_dropout: 注意力dropout概率
            activation: 激活函数类型（'gelu'/'relu'/'GLU'）
            query_cross_time: 是否跨时间步共享查询（保留参数以兼容原始接口）
            norm_flag: 归一化位置（'none'/'first'/'after'）
            return_att: 是否返回注意力权重
            num_prototypes: Memory Bank中原型数量（默认16）
            bypass_alpha: 残差绕过强度（0=关闭；如 0.1 表示 output = MAPA(x) + 0.1*x）
                目的：让原始输入 x 有一条跳过 MAPA 的 16→307 原型压缩的支路，
                缓解"信息瓶颈"。第一版固定 alpha 以干净验证机制；后续可改为可学习。
        """
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.query_cross_time = query_cross_time
        self.norm_flag = norm_flag
        self.return_att = return_att
        self.num_prototypes = num_prototypes
        self.bypass_alpha = float(bypass_alpha)
        
        # ===== Memory Bank（记忆库）=====
        # 可学习的原型向量集合，形状为 (num_prototypes, hid_dim)
        # 每个原型都是一个hid_dim维向量，通过训练学习到不同的交通模式
        self.memory_bank = nn.Parameter(torch.zeros(num_prototypes, hid_dim))
        nn.init.xavier_uniform_(self.memory_bank)
        
        # ===== 注意力层 =====
        # node2proxy：节点特征 -> 代理特征（用Memory Bank作为Query）
        # proxy2node：代理特征 -> 节点特征（用节点特征作为Query）
        self.node2proxy = AttentionLayer(hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att)
        self.proxy2node = AttentionLayer(hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att)
        
        self.dropout = nn.Dropout(dropout)
        
        # ===== 归一化层 =====
        if norm_flag=='first': 
            self.norm0 = nn.LayerNorm(hid_dim)
        if norm_flag != 'none':
            self.norm1 = nn.LayerNorm(hid_dim)
            self.norm2 = nn.LayerNorm(hid_dim)
        
        # ===== 激活函数 =====
        assert activation in ['gelu','relu','GLU']
        activation_func_dict={'gelu':nn.GELU(),
                              'relu':nn.ReLU(),
                              'GLU':nn.GLU()}
        if activation=='GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP2 = nn.Sequential(nn.Linear(hid_dim, d_ff1),
                                  activation_func_dict[activation],
                                  nn.Linear(d_ff, hid_dim))
    
    def forward(self, data):  # data: (B, T, N, C)
        """
        前向传播：记忆增强原型注意力
        
        参数：
            data: (B, T, N, hid_dim)  # 输入特征
        返回：
            final_out: (B, T, N, hid_dim)  # 输出特征
            ortho_loss: scalar tensor  # 正交损失
        """
        batch = data.shape[0]
        T = data.shape[1]
        
        # ===== 生成代理特征（使用Memory Bank）=====
        # Memory Bank: (num_prototypes, hid_dim)
        # 扩展到匹配batch维度：(B*T, num_prototypes, hid_dim)
        memory_bank_exp = self.memory_bank.unsqueeze(0).expand(batch * T, -1, -1)
        
        # 将输入数据展平：(B, T, N, hid_dim) -> (B*T, N, hid_dim)
        data_flat = rearrange(data, 'b t n c -> (b t) n c')
        
        # ===== 注意力计算 =====
        if self.norm_flag=='first':
            # 先归一化，再计算注意力
            z_proxy = self.norm0(memory_bank_exp)  # (B*T, num_prototypes, hid_dim)
            data_norm = self.norm1(data_flat)  # (B*T, N, hid_dim)
            
            # node2proxy：Memory Bank作为Query，节点特征作为Key/Value
            # 输出：(B*T, num_prototypes, hid_dim)
            proxy_feature, A1 = self.node2proxy(z_proxy, data_norm, data_norm)
            
            # proxy2node：节点特征作为Query，代理特征作为Key/Value
            # 输出：(B*T, N, hid_dim)
            node_feature, A2 = self.proxy2node(data_norm, proxy_feature, proxy_feature)
            
            # 残差连接 + 前馈网络
            enc_feature = data_flat + self.dropout(node_feature)
            enc_feature = self.norm2(enc_feature)
            enc_feature = enc_feature + self.dropout(self.MLP2(enc_feature))
        
        elif self.norm_flag == 'after':
            # 先计算注意力，再归一化
            proxy_feature, A1 = self.node2proxy(memory_bank_exp, data_flat, data_flat)
            node_feature, A2 = self.proxy2node(data_flat, proxy_feature, proxy_feature)
            enc_feature = data_flat + self.dropout(node_feature)
            enc_feature = self.norm1(enc_feature)
            enc_feature = enc_feature + self.dropout(self.MLP2(enc_feature))
            enc_feature = self.norm2(enc_feature)
        
        elif self.norm_flag == 'none':
            # 不归一化
            proxy_feature, A1 = self.node2proxy(memory_bank_exp, data_flat, data_flat)
            node_feature, A2 = self.proxy2node(data_flat, proxy_feature, proxy_feature)
            enc_feature = data_flat + self.dropout(node_feature)
            enc_feature = enc_feature + self.dropout(self.MLP2(enc_feature))
        
        # ===== 计算正交损失 =====
        # 目的：防止Memory Bank中的原型向量坍缩（collapse）为相同向量
        # 方法：计算原型之间的余弦相似度矩阵，将非对角线元素的平方和作为损失
        
        # 对Memory Bank进行L2归一化：(num_prototypes, hid_dim)
        memory_norm = F.normalize(self.memory_bank, dim=-1)
        
        # 计算余弦相似度矩阵：(num_prototypes, num_prototypes)
        # cosine_sim[i, j] = cosine_similarity(prototype[i], prototype[j])
        cosine_sim = torch.mm(memory_norm, memory_norm.t())
        
        # 构造掩码：非对角线元素为1，对角线元素为0
        mask = 1 - torch.eye(self.num_prototypes, device=data.device)
        
        # Normalize by the number of ordered prototype pairs, then scale to the
        # historical P=16 convention. This is exactly equal to the old sum at
        # P=16 while making lambda_ortho comparable for P=8/24/etc.
        pair_count = self.num_prototypes * (self.num_prototypes - 1)
        off_diagonal_sum = torch.sum((cosine_sim * mask) ** 2)
        if pair_count > 0:
            ortho_loss = off_diagonal_sum * (16 * 15) / pair_count
        else:
            ortho_loss = off_diagonal_sum
        
        # ===== 恢复原始形状 =====
        final_out = rearrange(enc_feature, '(b T) N d -> b T N d', b=batch)

        # ===== 残差绕过（ablation：验证 MAPA 压缩是否是瓶颈）=====
        # out = MAPA(x) + alpha * x
        # 第一版用固定 alpha 干净验证机制（避免学习 alpha → 1 退化）。后续可改 learnable。
        if self.bypass_alpha != 0.0:
            final_out = final_out + self.bypass_alpha * data

        # ===== 返回结果 =====
        if self.return_att:
            # 如果返回注意力权重，必须同时返回 ortho_loss 防止外层解包失败
            A_list = [A1, A2]
            return final_out, ortho_loss, A_list
        else:
            # 如果不返回注意力权重，返回 (final_out, ortho_loss)
            return final_out, ortho_loss


# ========================= AGCRN 式空间聚合层 =========================
# 用于替换 MAPA（记忆增强原型注意力）的空间模块。
# 关键区别：MAPA 把 307 个节点压成 16 个 prototype 再还原（丢节点级细节）；
# 本层用「节点可学习嵌入 -> 自适应邻接」做图卷积，保留全部节点粒度，
# 并融合 Pearson 静态先验图（AGCRN 的双邻接分支思想），用 GLU 门控做残差融合。
# 注意：先前证伪的是「AdaptiveGraphConv 加到 MAPA 上」(exp07041840)，
#       这里是「直接替换 MAPA」且模块更丰富（双分支 + GLU 门控 + LayerNorm），属不同集成。
# =============================================================================

class AGCRNSpatial(nn.Module):
    """
    AGCRN 式空间聚合层（替代 MAPA_EncoderLayer）。

    输入形状：data: (B, T, N, C)
    输出形状：     (B, T, N, C)
    """
    def __init__(self, hid_dim, num_nodes, dropout=0.1,
                 static_adj=None, use_static=True, activation='gelu'):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes

        # ===== 节点可学习嵌入（构建自适应邻接矩阵）=====
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, hid_dim))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, hid_dim))
        nn.init.xavier_uniform_(self.node_emb1)
        nn.init.xavier_uniform_(self.node_emb2)

        # ===== 自适应分支线性变换 =====
        self.W_adaptive = nn.Linear(hid_dim, hid_dim)

        # ===== 静态分支（可选，融合 Pearson 先验图）=====
        self.use_static = bool(use_static) and (static_adj is not None)
        if self.use_static:
            self.W_static = nn.Linear(hid_dim, hid_dim)
            self.register_buffer('static_adj', self._normalize_adj(static_adj))

        # ===== 双分支融合门（per-channel 凸组合，sigmoid 保证 [0,1]）=====
        self.branch_gate = nn.Parameter(torch.zeros(hid_dim))

        # ===== GLU 残差门控（聚合特征与输入融合）=====
        self.gate_linear = nn.Linear(hid_dim * 2, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def _normalize_adj(self, A):
        A = F.relu(A)
        return F.softmax(A, dim=1)

    def _adaptive_adj(self):
        A = F.softmax(F.relu(self.node_emb1 @ self.node_emb2.T), dim=1)
        return A

    def forward(self, data):  # data: (B, T, N, C)
        B, T, N, C = data.shape
        x_flat = data.reshape(B * T, N, C)

        # 自适应分支：A_adaptive @ X -> Linear -> 激活
        A_adapt = self._adaptive_adj()  # (N, N)
        out_adapt = torch.einsum('nm,bmc->bnc', A_adapt, x_flat)
        out_adapt = self.activation(self.W_adaptive(out_adapt))

        # 静态分支（Pearson 先验）
        if self.use_static:
            out_static = torch.einsum('nm,bmc->bnc', self.static_adj, x_flat)
            out_static = self.activation(self.W_static(out_static))
            g = torch.sigmoid(self.branch_gate)
            agg = g * out_adapt + (1 - g) * out_static
        else:
            agg = out_adapt

        agg = self.dropout(agg)

        # GLU 残差门控：gate = sigmoid(W·[agg, x])，out = gate*agg + (1-gate)*x
        gate_in = torch.cat([agg, x_flat], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_in))
        out = gate * agg + (1 - gate) * x_flat
        out = self.norm(out)

        out = out.reshape(B, T, N, C)
        return out


# ========================= 改进的 Model 类（第二部分）=========================
# 展示如何将 MS_TCN、FTLE、MAPA_EncoderLayer 三个模块串联
# ================================================================================


class VolatilityConditionedDeltaScale(nn.Module):
    """Scale forecast deltas using only observable historical volatility.

    The final projection is zero-initialized, so the initial scale is exactly
    one and enabling this module initially leaves every forecast unchanged.
    """

    preserve_initialization = True

    def __init__(self, predict_length, hidden_dim=8, max_adjustment=0.1):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError('volatility_delta_hidden must be positive')
        if not 0.0 < max_adjustment <= 1.0:
            raise ValueError(
                'volatility_delta_max must be in the interval (0, 1]'
            )
        self.predict_length = int(predict_length)
        self.max_adjustment = float(max_adjustment)
        self.scale_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.predict_length),
        )
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)
        self.last_scale = None

    def forward(self, prediction, last_value, history):
        if history.shape[1] < 2:
            raise ValueError(
                'Volatility-conditioned delta scale requires at least 2 '
                'historical steps'
            )
        # PEMS uses the first channel as the forecast target. Global z-score
        # scaling multiplies volatility by one constant, preserving regimes.
        flow = history[..., 0]
        volatility = (flow[:, 1:] - flow[:, :-1]).abs().mean(dim=1)
        feature = torch.log1p(volatility).unsqueeze(-1)  # (B, N, 1)
        logits = self.scale_net(feature)                 # (B, N, H)
        scale = 1.0 + self.max_adjustment * torch.tanh(logits)
        scale = scale.permute(0, 2, 1).unsqueeze(-1)     # (B, H, N, 1)
        self.last_scale = scale.detach()
        return last_value + scale * (prediction - last_value)


class SeasonalDriftResidualAdapter(nn.Module):
    """Add a training-only TOD seasonal-drift correction to a parent forecast.

    The correction uses no traffic value after the input cutoff.  It is the
    difference between the training-split average flow at the known future
    time-of-day and at the last observed time-of-day for the same sensor.  A
    single learnable coefficient per forecast horizon controls that fixed
    pattern.  The coefficients are initialized to zero, so this branch is
    exactly an identity mapping before fitting and can be safely trained on a
    frozen parent checkpoint.
    """

    preserve_initialization = True

    def __init__(self, predict_length, max_scale=1.0):
        super().__init__()
        if predict_length <= 0:
            raise ValueError('predict_length must be positive')
        if not 0.0 < max_scale <= 1.0:
            raise ValueError(
                'seasonal_drift_max must be in the interval (0, 1]'
            )
        self.predict_length = int(predict_length)
        self.max_scale = float(max_scale)
        # tanh(0) = 0: at epoch 0 the loaded parent output is unchanged.
        self.horizon_logits = nn.Parameter(torch.zeros(self.predict_length))
        self.last_horizon_scale = None
        self.last_correction = None

    def forward(self, prediction, seasonal_mean, x_time, y_time):
        if prediction.ndim != 4:
            raise ValueError('prediction must be shaped (B,H,N,C)')
        if prediction.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} forecast steps, got '
                f'{prediction.shape[1]}'
            )
        if prediction.shape[-1] != 1:
            raise ValueError(
                'SeasonalDriftResidualAdapter currently supports pre_dim=1'
            )
        if seasonal_mean.ndim != 2:
            raise ValueError(
                'seasonal_mean must be shaped (time_of_day, num_nodes)'
            )
        if seasonal_mean.shape[1] != prediction.shape[2]:
            raise ValueError(
                'seasonal_mean node dimension must match prediction'
            )
        if x_time is None or y_time is None:
            raise ValueError(
                'SeasonalDriftResidualAdapter requires input and future '
                'time-of-day indices'
            )
        if x_time.ndim != 3 or y_time.ndim != 3:
            raise ValueError('x_time and y_time must be shaped (B,T,2)')
        if y_time.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} future time steps, got '
                f'{y_time.shape[1]}'
            )

        num_tod = seasonal_mean.shape[0]
        last_tod = x_time[:, -1, 0].long().clamp(0, num_tod - 1)
        future_tod = y_time[:, :, 0].long().clamp(0, num_tod - 1)
        last_mean = seasonal_mean[last_tod].unsqueeze(1)       # (B,1,N)
        future_mean = seasonal_mean[future_tod]                # (B,H,N)
        seasonal_delta = future_mean - last_mean
        horizon_scale = self.max_scale * torch.tanh(
            self.horizon_logits
        ).view(1, self.predict_length, 1, 1)
        correction = horizon_scale * seasonal_delta.unsqueeze(-1)
        self.last_horizon_scale = horizon_scale.detach()
        self.last_correction = correction.detach()
        return prediction + correction


class AnalogRetrievalResidualAdapter(nn.Module):
    """Blend a frozen parent forecast toward a train-memory analog forecast.

    ``retrieval`` is precomputed using only training-split candidate windows.
    Training queries exclude their own window, while validation and test
    queries search the same fixed training memory.  One bounded coefficient
    per horizon is learned on the training targets.  Zero initialization makes
    the complete model exactly equal to its parent before the first update.
    """

    preserve_initialization = True

    def __init__(self, predict_length, max_blend=1.0):
        super().__init__()
        if predict_length <= 0:
            raise ValueError('predict_length must be positive')
        if not 0.0 < max_blend <= 1.0:
            raise ValueError(
                'analog_retrieval_max must be in the interval (0, 1]'
            )
        self.predict_length = int(predict_length)
        self.max_blend = float(max_blend)
        self.horizon_logits = nn.Parameter(torch.zeros(self.predict_length))
        self.last_horizon_blend = None
        self.last_correction = None

    def forward(self, prediction, retrieval):
        if prediction.ndim != 4 or retrieval.ndim != 4:
            raise ValueError(
                'prediction and retrieval must be shaped (B,H,N,C)'
            )
        if prediction.shape != retrieval.shape:
            raise ValueError(
                'prediction and retrieval shapes must match exactly: '
                f'{tuple(prediction.shape)} != {tuple(retrieval.shape)}'
            )
        if prediction.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} forecast steps, got '
                f'{prediction.shape[1]}'
            )
        if not torch.isfinite(retrieval).all():
            raise ValueError('retrieval contains non-finite values')

        horizon_blend = self.max_blend * torch.tanh(
            self.horizon_logits
        ).view(1, self.predict_length, 1, 1)
        correction = horizon_blend * (retrieval - prediction)
        self.last_horizon_blend = horizon_blend.detach()
        self.last_correction = correction.detach()
        return prediction + correction


class AnalogLowRankRetrievalResidualAdapter(nn.Module):
    """Node/horizon-specific analog blend with a low-rank constraint.

    A shared horizon logit models the global retrieval contribution.  The
    product of a node factor and a horizon factor captures persistent sensor
    reliability without learning an unconstrained N-by-H coefficient table.
    Node factors start at exactly zero, so the complete branch is an identity
    mapping at epoch 0 even though the horizon factor has a small nonzero
    initialization that permits immediate gradients into the node factors.
    """

    preserve_initialization = True

    def __init__(
        self,
        num_nodes,
        predict_length,
        rank=2,
        max_blend=1.0,
    ):
        super().__init__()
        if num_nodes <= 0 or predict_length <= 0:
            raise ValueError('num_nodes and predict_length must be positive')
        if rank <= 0 or rank > min(num_nodes, predict_length):
            raise ValueError(
                'analog_lowrank_rank must be positive and no larger than '
                'min(num_nodes, predict_length)'
            )
        if not 0.0 < max_blend <= 1.0:
            raise ValueError(
                'analog_retrieval_max must be in the interval (0, 1]'
            )
        self.num_nodes = int(num_nodes)
        self.predict_length = int(predict_length)
        self.rank = int(rank)
        self.max_blend = float(max_blend)
        self.horizon_logits = nn.Parameter(torch.zeros(self.predict_length))
        self.node_factor = nn.Parameter(torch.zeros(
            self.num_nodes, self.rank
        ))
        self.horizon_factor = nn.Parameter(torch.empty(
            self.predict_length, self.rank
        ))
        nn.init.normal_(self.horizon_factor, mean=0.0, std=0.02)
        self.last_blend = None
        self.last_correction = None

    def forward(self, prediction, retrieval):
        if prediction.ndim != 4 or retrieval.ndim != 4:
            raise ValueError(
                'prediction and retrieval must be shaped (B,H,N,C)'
            )
        if prediction.shape != retrieval.shape:
            raise ValueError(
                'prediction and retrieval shapes must match exactly: '
                f'{tuple(prediction.shape)} != {tuple(retrieval.shape)}'
            )
        if prediction.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} forecast steps, got '
                f'{prediction.shape[1]}'
            )
        if prediction.shape[2] != self.num_nodes:
            raise ValueError(
                f'Expected {self.num_nodes} nodes, got '
                f'{prediction.shape[2]}'
            )
        if not torch.isfinite(retrieval).all():
            raise ValueError('retrieval contains non-finite values')

        lowrank_logits = torch.einsum(
            'nr,hr->hn', self.node_factor, self.horizon_factor
        )
        blend_logits = self.horizon_logits[:, None] + lowrank_logits
        blend = self.max_blend * torch.tanh(blend_logits)
        blend = blend.view(1, self.predict_length, self.num_nodes, 1)
        correction = blend * (retrieval - prediction)
        self.last_blend = blend.detach()
        self.last_correction = correction.detach()
        return prediction + correction


class AnalogTemporalLowRankDecoder(nn.Module):
    """Learn a tiny low-rank temporal correction for analog retrievals.

    The compressed cache is decoded with a fixed train-only PCA basis before
    it reaches the model. This module adds a supervised, shared rank-``r``
    temporal correction to that decoded near-match sequence. It operates on
    each node/channel independently, so its parameter count is only
    ``2 * horizon * rank`` and does not grow with the sensor count.

    The encoder starts with a small random value while the decoder is exactly
    zero. Consequently the complete module is an identity mapping at epoch
    0, but gradients can immediately reach the decoder. Exact-history memory
    is deliberately outside this module and remains lossless and unchanged.
    """

    preserve_initialization = True

    def __init__(self, predict_length, rank=4, max_scale=0.25):
        super().__init__()
        if predict_length <= 0:
            raise ValueError('predict_length must be positive')
        if rank <= 0 or rank > predict_length:
            raise ValueError(
                'analog_temporal_decoder_rank must be in [1, predict_length]'
            )
        if not 0.0 < max_scale <= 1.0:
            raise ValueError(
                'analog_temporal_decoder_max_scale must be in (0, 1]'
            )
        self.predict_length = int(predict_length)
        self.rank = int(rank)
        self.max_scale = float(max_scale)
        self.encoder = nn.Parameter(torch.empty(
            self.predict_length, self.rank
        ))
        self.decoder = nn.Parameter(torch.zeros(
            self.predict_length, self.rank
        ))
        nn.init.normal_(self.encoder, mean=0.0, std=0.02)
        self.last_correction = None

    def forward(self, retrieval):
        if retrieval.ndim != 4:
            raise ValueError('retrieval must be shaped (B,H,N,C)')
        if retrieval.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} retrieval steps, got '
                f'{retrieval.shape[1]}'
            )
        if not torch.isfinite(retrieval).all():
            raise ValueError('retrieval contains non-finite values')

        # Remove the per-sample/node level so the low-rank path learns only a
        # temporal-shape correction and cannot introduce an arbitrary offset.
        centered = retrieval - retrieval.mean(dim=1, keepdim=True)
        code = torch.einsum('bhnc,hr->brnc', centered, self.encoder)
        correction = torch.einsum('brnc,hr->bhnc', code, self.decoder)
        correction = self.max_scale * correction / float(self.rank) ** 0.5
        self.last_correction = correction.detach()
        return retrieval + correction


class AnalogTemporalGapResidualAdapter(nn.Module):
    """Cross-horizon low-rank correction after all retrieval refinements.

    ``reference`` is assembled causally from train-memory continuations:
    exact-history memory where available and the reranked near-match memory
    elsewhere. A shared temporal encoder/decoder captures cross-horizon error
    structure, while a rank-wise node gate adds sensor specificity with only
    ``N * rank`` extra parameters. The zero decoder makes the complete branch
    an exact identity at initialization.
    """

    preserve_initialization = True

    def __init__(
        self,
        num_nodes,
        predict_length,
        rank=4,
        max_scale=0.25,
    ):
        super().__init__()
        if num_nodes <= 0 or predict_length <= 0:
            raise ValueError('num_nodes and predict_length must be positive')
        if rank <= 0 or rank > predict_length:
            raise ValueError(
                'analog_temporal_gap_rank must be in [1, predict_length]'
            )
        if not 0.0 < max_scale <= 1.0:
            raise ValueError('analog_temporal_gap_max must be in (0, 1]')
        self.num_nodes = int(num_nodes)
        self.predict_length = int(predict_length)
        self.rank = int(rank)
        self.max_scale = float(max_scale)
        self.encoder = nn.Parameter(torch.empty(
            self.predict_length, self.rank
        ))
        self.decoder = nn.Parameter(torch.zeros(
            self.predict_length, self.rank
        ))
        self.node_factor = nn.Parameter(torch.zeros(
            self.num_nodes, self.rank
        ))
        nn.init.normal_(self.encoder, mean=0.0, std=0.02)
        self.last_correction = None

    def forward(self, prediction, reference):
        if prediction.ndim != 4 or reference.ndim != 4:
            raise ValueError(
                'prediction and reference must be shaped (B,H,N,C)'
            )
        if prediction.shape != reference.shape:
            raise ValueError(
                'prediction and reference shapes must match exactly: '
                f'{tuple(prediction.shape)} != {tuple(reference.shape)}'
            )
        if prediction.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} forecast steps, got '
                f'{prediction.shape[1]}'
            )
        if prediction.shape[2] != self.num_nodes:
            raise ValueError(
                f'Expected {self.num_nodes} nodes, got '
                f'{prediction.shape[2]}'
            )
        if not torch.isfinite(reference).all():
            raise ValueError('reference contains non-finite values')

        gap = reference - prediction
        code = torch.einsum('bhnc,hr->brnc', gap, self.encoder)
        node_gain = 1.0 + torch.tanh(self.node_factor)
        code = code * node_gain.transpose(0, 1)[None, :, :, None]
        correction = torch.einsum('brnc,hr->bhnc', code, self.decoder)
        correction = self.max_scale * correction / float(self.rank) ** 0.5
        self.last_correction = correction.detach()
        return prediction + correction


class AnalogQueryConditionedResidual(nn.Module):
    """Small causal conditioner that changes candidate residual content.

    Candidate-history difference summaries are observable at inference.  The
    output is zero-initialized so a frozen-parent run is an exact anchor.
    This module is intentionally defined for full-future candidate payloads;
    the code-domain path remains a separate storage/runtime experiment.
    """

    preserve_initialization = True

    def __init__(self, num_nodes, predict_length, feature_dim=8,
                 hidden_dim=8, max_adjustment=0.25):
        super().__init__()
        if num_nodes <= 0 or predict_length <= 0:
            raise ValueError('num_nodes and predict_length must be positive')
        if feature_dim <= 0 or hidden_dim <= 0 or max_adjustment <= 0:
            raise ValueError('invalid query-conditioned residual dimensions')
        self.num_nodes = int(num_nodes)
        self.predict_length = int(predict_length)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_adjustment = float(max_adjustment)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.horizon_embedding = nn.Parameter(torch.empty(predict_length, hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
        nn.init.normal_(self.node_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, candidate_future, candidate_available, candidate_features):
        if candidate_future.ndim != 5 or candidate_available.ndim != 5:
            raise ValueError('candidate tensors must be 5-dimensional')
        if candidate_future.shape != candidate_available.shape:
            raise ValueError('candidate future and availability must match')
        if candidate_features.ndim != 4:
            raise ValueError('candidate_features must be (B,K,N,F)')
        batch, candidates, horizon, nodes, channels = candidate_future.shape
        if (horizon, nodes) != (self.predict_length, self.num_nodes):
            raise ValueError('candidate future dimensions do not match conditioner')
        if tuple(candidate_features.shape) != (
            batch, candidates, nodes, self.feature_dim
        ):
            raise ValueError('candidate feature shape does not match conditioner')
        dtype = candidate_future.dtype
        features = candidate_features.to(dtype=dtype)
        hidden = self.feature_projection(features)[:, :, None, :, :]
        hidden = hidden + self.node_embedding.to(dtype=dtype).view(
            1, 1, 1, nodes, self.hidden_dim
        ) + self.horizon_embedding.to(dtype=dtype).view(
            1, 1, horizon, 1, self.hidden_dim
        )
        delta = self.max_adjustment * torch.tanh(
            self.output_projection(F.gelu(hidden))
        )
        return candidate_future + delta * candidate_available.to(dtype=dtype)


class AnalogCandidateHorizonReranker(nn.Module):
    """Reweight the fixed K train-memory continuations for every horizon.

    Candidate identities are selected using observable histories only.  This
    module sees compact history-comparison features and each candidate
    continuation's disagreement with the *frozen final parent* forecast.  It
    never consumes the query target.  The output projection is initialized to
    exactly zero; therefore the initial candidate weights are exactly uniform
    over available continuations and ``mean_retrieval`` is returned bit for
    bit, rather than recomputed from the lower-precision candidate tensor.
    """

    preserve_initialization = True

    def __init__(
        self,
        num_nodes,
        predict_length,
        feature_dim=8,
        disagreement_dim=6,
        hidden_dim=16,
        code_domain=False,
        code_rank=None,
        temporal_mean=None,
        temporal_basis=None,
        decode_candidates=True,
        level_align=False,
        candidate_parent_residual_features=False,
    ):
        super().__init__()
        if num_nodes <= 0 or predict_length <= 0:
            raise ValueError('num_nodes and predict_length must be positive')
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError('feature_dim and hidden_dim must be positive')
        if disagreement_dim <= 0:
            raise ValueError('disagreement_dim must be positive')
        self.num_nodes = int(num_nodes)
        self.predict_length = int(predict_length)
        self.feature_dim = int(feature_dim)
        self.disagreement_dim = int(disagreement_dim)
        self.hidden_dim = int(hidden_dim)
        self.code_domain = bool(code_domain)
        self.code_rank = int(code_rank or 0)
        self.decode_candidates = bool(decode_candidates)
        self.level_align = bool(level_align)
        self.candidate_parent_residual_features = bool(
            candidate_parent_residual_features
        )
        if self.candidate_parent_residual_features:
            if self.code_domain:
                raise ValueError(
                    'Candidate-parent residual features require full '
                    'candidate futures'
                )
            if self.disagreement_dim != 6:
                raise ValueError(
                    'Candidate-parent residual features require '
                    'disagreement_dim=6'
                )
        if self.level_align and self.code_domain:
            raise ValueError(
                'Level-aligned residual construction currently requires '
                'full-residual candidates'
            )
        if self.code_domain:
            if self.code_rank <= 0 or temporal_mean is None or temporal_basis is None:
                raise ValueError('code-domain reranker requires code_rank and PCA mean/basis')
            mean_tensor = torch.as_tensor(temporal_mean, dtype=torch.float32)
            basis_tensor = torch.as_tensor(temporal_basis, dtype=torch.float32)
            if tuple(mean_tensor.shape) != (self.predict_length, self.num_nodes, 1):
                raise ValueError(f'Unexpected PCA mean shape {tuple(mean_tensor.shape)}')
            if tuple(basis_tensor.shape) != (self.num_nodes, self.predict_length, self.code_rank):
                raise ValueError(f'Unexpected PCA basis shape {tuple(basis_tensor.shape)}')
            self.register_buffer('temporal_mean', mean_tensor, persistent=False)
            self.register_buffer('temporal_basis', basis_tensor, persistent=False)

        self.history_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.disagreement_projection = nn.Linear(
            self.disagreement_dim, self.hidden_dim
        )
        self.node_embedding = nn.Parameter(torch.empty(
            self.num_nodes, self.hidden_dim
        ))
        self.horizon_embedding = nn.Parameter(torch.empty(
            self.predict_length, self.hidden_dim
        ))
        self.output_projection = nn.Linear(self.hidden_dim, 1)

        nn.init.xavier_uniform_(self.history_projection.weight)
        nn.init.zeros_(self.history_projection.bias)
        nn.init.xavier_uniform_(self.disagreement_projection.weight)
        nn.init.zeros_(self.disagreement_projection.bias)
        nn.init.normal_(self.node_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        self.last_weights = None
        self.last_correction = None
        # Training-only tensors used by the optional utility-ranking loss.
        # ``last_logits`` deliberately keeps its graph; the remaining inputs
        # are detached because utility labels must not update the frozen
        # parent or candidate memory.
        self.last_logits = None
        self.last_parent_prediction = None
        self.last_candidate_future = None
        self.last_candidate_available = None
        self.last_code_decode_seconds = 0.0
        self.last_code_aggregate_seconds = 0.0
        self.last_code_final_decode_seconds = 0.0
        self.last_code_candidate_decode_count = 0
        self.last_code_final_decode_count = 0

    def forward(
        self,
        parent_prediction,
        mean_retrieval,
        candidate_future,
        candidate_available,
        candidate_features,
    ):
        if parent_prediction.ndim != 4 or mean_retrieval.ndim != 4:
            raise ValueError(
                'parent_prediction and mean_retrieval must be (B,H,N,C)'
            )
        if parent_prediction.shape != mean_retrieval.shape:
            raise ValueError(
                'parent_prediction and mean_retrieval must have equal shapes'
            )
        if candidate_future.ndim != 5 or candidate_available.ndim != 5:
            raise ValueError('Candidate tensors must be 5-dimensional')
        if self.code_domain:
            if candidate_future.shape[2] != self.code_rank:
                raise ValueError(
                    f'Code rank {candidate_future.shape[2]} != {self.code_rank}'
                )
            if candidate_available.shape[2] != self.predict_length:
                raise ValueError('Code candidate availability must retain H axis')
            if (
                candidate_future.shape[0] != candidate_available.shape[0]
                or candidate_future.shape[1] != candidate_available.shape[1]
                or candidate_future.shape[3:] != candidate_available.shape[3:]
            ):
                raise ValueError('Code candidate and availability axes do not match')
        elif self.candidate_parent_residual_features:
            if (
                candidate_future.shape[:-1] != candidate_available.shape[:-1]
                or candidate_future.shape[-1] != 2
                or candidate_available.shape[-1] != 1
            ):
                raise ValueError(
                    'Candidate-parent payload must be (B,K,H,N,2) and '
                    'availability must be (B,K,H,N,1)'
                )
        elif candidate_future.shape != candidate_available.shape:
            raise ValueError('Candidate future and availability must match')
        if candidate_features.ndim != 4:
            raise ValueError(
                'candidate_features must be shaped (B,K,N,F)'
            )

        batch, candidates, horizon_or_rank, nodes, payload_channels = candidate_future.shape
        horizon = self.predict_length if self.code_domain else horizon_or_rank
        channels = parent_prediction.shape[-1]
        if (
            tuple(parent_prediction.shape)
            != (batch, horizon, nodes, channels)
        ):
            raise ValueError(
                'Candidate future axes do not match the parent prediction'
            )
        if horizon != self.predict_length or nodes != self.num_nodes:
            raise ValueError(
                f'Expected H={self.predict_length}, N={self.num_nodes}; '
                f'got H={horizon}, N={nodes}'
            )
        if tuple(candidate_features.shape) != (
            batch, candidates, nodes, self.feature_dim
        ):
            raise ValueError(
                'Unexpected candidate feature shape: '
                f'{tuple(candidate_features.shape)}'
            )
        if not (
            torch.isfinite(mean_retrieval).all()
            and torch.isfinite(candidate_future).all()
            and torch.isfinite(candidate_features).all()
        ):
            raise ValueError('Candidate reranker inputs contain non-finite values')

        dtype = parent_prediction.dtype
        candidate_payload = candidate_future.to(dtype=dtype)
        candidate_features = candidate_features.to(dtype=dtype)
        available = candidate_available.to(dtype=dtype)
        available = (available > 0).to(dtype=dtype)
        level_delta = None
        if self.level_align:
            # candidate_features[..., 2] is tanh(last query-history minus
            # candidate-history difference, divided by 3).  Invert this
            # bounded encoding to recover the observable z-score level shift.
            # The shift is applied after the original scorer has produced its
            # weights, so the frozen reranker sees exactly its trained input
            # distribution.
            encoded_level_delta = candidate_features[..., 2].clamp(-0.999999, 0.999999)
            level_delta = 3.0 * torch.atanh(encoded_level_delta)
        self.last_code_decode_seconds = 0.0
        self.last_code_aggregate_seconds = 0.0
        self.last_code_final_decode_seconds = 0.0
        self.last_code_candidate_decode_count = 0
        self.last_code_final_decode_count = 0

        if self.code_domain and self.decode_candidates:
            decode_start = time.perf_counter()
            candidate_future = self.temporal_mean.to(dtype=dtype)[None, None] + torch.einsum(
                'bkrnc,nhr->bkhnc', candidate_payload,
                self.temporal_basis.to(dtype=dtype),
            )
            self.last_code_decode_seconds = time.perf_counter() - decode_start
            self.last_code_candidate_decode_count = int(candidates)
            available_for_logits = available.all(dim=2, keepdim=True).expand(
                -1, -1, self.predict_length, -1, -1
            )
        elif self.code_domain:
            candidate_future = None
            self.last_code_decode_seconds = 0.0
            available_for_logits = available.all(dim=2, keepdim=True).expand(
                -1, -1, self.predict_length, -1, -1
            )
        else:
            if self.candidate_parent_residual_features:
                candidate_future = candidate_payload[..., :channels]
                candidate_parent_prediction = candidate_payload[
                    ..., channels:(2 * channels)
                ]
            else:
                candidate_future = candidate_payload
                candidate_parent_prediction = None
            available_for_logits = available

        history_hidden = self.history_projection(
            candidate_features
        )[:, :, None, :, :]
        node_hidden = self.node_embedding.view(
            1, 1, 1, self.num_nodes, self.hidden_dim
        )
        horizon_hidden = self.horizon_embedding.view(
            1, 1, self.predict_length, 1, self.hidden_dim
        )
        if self.code_domain and not self.decode_candidates:
            hidden = F.gelu(history_hidden + node_hidden + horizon_hidden)
        else:
            parent = parent_prediction[:, None, :, :, :]
            signed_disagreement = candidate_future - parent
            abs_disagreement = signed_disagreement.abs()
            squared_disagreement = signed_disagreement * signed_disagreement
            max_abs = abs_disagreement.amax(dim=1, keepdim=True).clamp_min(1e-6)
            normalized_signed = signed_disagreement / max_abs
            parent_abs = parent.abs().clamp_min(1e-3)
            relative_signed = signed_disagreement / parent_abs
            relative_abs = abs_disagreement / parent_abs
            if self.candidate_parent_residual_features:
                candidate_residual = (
                    candidate_future - candidate_parent_prediction
                )
                parent_delta = parent - candidate_parent_prediction
                disagreement_features = torch.cat([
                    signed_disagreement,
                    abs_disagreement,
                    candidate_residual,
                    candidate_residual.abs(),
                    parent_delta,
                    parent_delta.abs(),
                ], dim=-1)
            elif self.disagreement_dim == 2:
                disagreement_features = torch.cat([signed_disagreement, abs_disagreement], dim=-1)
            elif self.disagreement_dim == 6:
                disagreement_features = torch.cat([
                    signed_disagreement, abs_disagreement, squared_disagreement,
                    normalized_signed, relative_signed, relative_abs,
                ], dim=-1)
            else:
                raise ValueError(
                    'Unsupported disagreement_dim '
                    f'{self.disagreement_dim}; expected 2 (legacy) or 6 (rich).'
                )
            disagreement_hidden = self.disagreement_projection(disagreement_features)
            hidden = F.gelu(history_hidden + disagreement_hidden + node_hidden + horizon_hidden)
        logits = self.output_projection(hidden)

        # Numerically stable masked softmax.  At initialization logits are
        # exactly zero.  Both ``weights`` and ``uniform`` then execute the same
        # division, so their difference is exactly zero for every valid mask.
        mask = available_for_logits
        masked_logits = torch.where(
            mask > 0, logits, torch.full_like(logits, -1e9)
        )
        max_logit = masked_logits.max(dim=1, keepdim=True).values
        exponential = torch.exp(masked_logits - max_logit) * mask
        available_count = mask.sum(dim=1, keepdim=True)
        weights = exponential / available_count.new_tensor(1.0).maximum(
            exponential.sum(dim=1, keepdim=True)
        )
        uniform = mask / available_count.clamp_min(1.0)
        weight_delta = weights - uniform
        if self.code_domain:
            # Temporal codes need one candidate weight per node.  Collapse the
            # horizon-specific scorer weights before code aggregation.
            aggregate_start = time.perf_counter()
            node_weights = weights.mean(dim=2)
            node_weights = node_weights / node_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            aggregate_code = (
                node_weights.unsqueeze(2) * candidate_payload
            ).sum(dim=1)
            self.last_code_aggregate_seconds = time.perf_counter() - aggregate_start
            final_decode_start = time.perf_counter()
            correction = self.temporal_mean.to(dtype=dtype) + torch.einsum(
                'brnc,nhr->bhnc', aggregate_code,
                self.temporal_basis.to(dtype=dtype),
            )
            self.last_code_final_decode_seconds = (
                time.perf_counter() - final_decode_start
            )
            self.last_code_final_decode_count = 1
            self.last_weights = node_weights.detach()
        else:
            correction = (weight_delta * candidate_future).sum(dim=1)
            if level_delta is not None:
                correction = correction + (
                    weights * level_delta[:, :, None, :, None]
                ).sum(dim=1)
            self.last_weights = weights.detach()
        self.last_correction = correction.detach()
        self.last_logits = logits
        self.last_parent_prediction = parent_prediction.detach()
        self.last_candidate_future = (
            None if candidate_future is None else candidate_future.detach()
        )
        self.last_candidate_available = mask.detach()
        return correction if self.code_domain else mean_retrieval + correction

    def utility_ranking_loss(
        self,
        target,
        target_valid,
        temperature=1.0,
    ):
        """Listwise supervision from train-target correction utility.

        Utility is computed in the normalized forecast space as the parent
        absolute error minus the candidate absolute error.  Utilities are
        standardized across the K candidates independently for every
        (batch,horizon,node,channel), so the temperature has a stable meaning
        across traffic scales.  Targets are detached; gradients flow only
        through the reranker logits.
        """
        if float(temperature) <= 0.0:
            raise ValueError('utility ranking temperature must be positive')
        if any(value is None for value in (
            self.last_logits,
            self.last_parent_prediction,
            self.last_candidate_future,
            self.last_candidate_available,
        )):
            raise RuntimeError(
                'utility_ranking_loss requires a preceding reranker forward'
            )
        if target.ndim != 4 or target_valid.ndim != 4:
            raise ValueError('target and target_valid must be (B,H,N,C)')
        if target.shape != self.last_parent_prediction.shape:
            raise ValueError(
                'utility target does not match the cached parent prediction'
            )

        logits = self.last_logits
        parent = self.last_parent_prediction[:, None]
        candidate = self.last_candidate_future
        available = self.last_candidate_available > 0
        valid = available & target_valid[:, None].bool()
        valid_float = valid.to(dtype=logits.dtype)
        available_count = valid_float.sum(dim=1, keepdim=True)

        with torch.no_grad():
            utility = (
                (parent - target[:, None]).abs()
                - (candidate - target[:, None]).abs()
            )
            utility_sum = (utility * valid_float).sum(dim=1, keepdim=True)
            utility_mean = utility_sum / available_count.clamp_min(1.0)
            centered = (utility - utility_mean) * valid_float
            utility_variance = (
                centered.square().sum(dim=1, keepdim=True)
                / available_count.clamp_min(1.0)
            )
            standardized = centered / utility_variance.add(1e-6).sqrt()
            target_logits = standardized / float(temperature)
            target_logits = torch.where(
                valid,
                target_logits,
                torch.full_like(target_logits, -1e9),
            )
            target_weights = torch.softmax(target_logits, dim=1)
            target_weights = target_weights * valid_float
            target_weights = target_weights / target_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)

        masked_logits = torch.where(
            valid, logits, torch.full_like(logits, -1e9)
        )
        log_probabilities = torch.log_softmax(masked_logits, dim=1)
        per_position = -(target_weights * log_probabilities).sum(dim=1)
        position_valid = (
            target_valid.bool()
            & (available_count.squeeze(1) >= 2)
        )
        if not bool(position_valid.any()):
            return logits.sum() * 0.0, {
                'valid_positions': 0,
                'target_best_probability': 0.0,
            }
        loss = per_position[position_valid].mean()
        target_best_probability = target_weights.max(dim=1).values[
            position_valid
        ].mean()
        return loss, {
            'valid_positions': int(position_valid.sum().item()),
            'target_best_probability': float(
                target_best_probability.detach().item()
            ),
        }


class AnalogCorrectionUtilityGate(nn.Module):
    """Prediction-aware control of the frozen near-match correction.

    Positive outputs suppress the frozen low-rank correction and negative
    outputs strengthen it.  Only inference-observable signals are used.  The
    final projection is zero initialized, so the initial output exactly
    matches ``near_prediction``.
    """

    preserve_initialization = True

    def __init__(
        self,
        num_nodes,
        predict_length,
        feature_dim=8,
        hidden_dim=8,
        node_rank=2,
        max_suppression=1.0,
    ):
        super().__init__()
        if num_nodes <= 0 or predict_length <= 0:
            raise ValueError('num_nodes and predict_length must be positive')
        if feature_dim != 8:
            raise ValueError('correction utility gate currently requires 8 features')
        if hidden_dim <= 0 or node_rank <= 0:
            raise ValueError('hidden_dim and node_rank must be positive')
        if max_suppression <= 0:
            raise ValueError('max_suppression must be positive')

        self.num_nodes = int(num_nodes)
        self.predict_length = int(predict_length)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.node_rank = int(node_rank)
        self.max_suppression = float(max_suppression)

        self.feature_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.horizon_embedding = nn.Parameter(torch.empty(
            self.predict_length, self.hidden_dim
        ))
        self.node_factor = nn.Parameter(torch.empty(
            self.num_nodes, self.node_rank
        ))
        self.node_projection = nn.Parameter(torch.empty(
            self.node_rank, self.hidden_dim
        ))
        self.output_projection = nn.Linear(self.hidden_dim, 1)

        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.node_factor, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.node_projection)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        self.last_suppression = None
        self.last_correction = None

    def forward(
        self,
        base_prediction,
        near_prediction,
        retrieval_prediction,
        quality_retrieval,
        reranker_weights,
    ):
        if any(value.ndim != 4 for value in (
            base_prediction, near_prediction, retrieval_prediction,
        )):
            raise ValueError('Prediction inputs must be shaped (B,H,N,C)')
        if not (
            base_prediction.shape == near_prediction.shape
            == retrieval_prediction.shape
        ):
            raise ValueError('Prediction inputs must have identical shapes')
        batch, horizon, nodes, channels = base_prediction.shape
        if horizon != self.predict_length or nodes != self.num_nodes:
            raise ValueError(
                f'Expected H={self.predict_length}, N={self.num_nodes}; '
                f'got H={horizon}, N={nodes}'
            )
        if channels != 1:
            raise ValueError('correction utility gate currently supports C=1')
        if quality_retrieval is None or quality_retrieval.ndim != 4:
            raise ValueError('quality_retrieval must be shaped (B,H,N,C+2)')
        if quality_retrieval.shape[:3] != (batch, horizon, nodes):
            raise ValueError('quality_retrieval axes do not match predictions')
        if quality_retrieval.shape[-1] < channels + 2:
            raise ValueError(
                'correction utility gate requires distance and dispersion channels'
            )
        if reranker_weights is None or reranker_weights.ndim != 5:
            raise ValueError('reranker_weights must be shaped (B,K,H,N,C)')
        if reranker_weights.shape[0] != batch or tuple(
            reranker_weights.shape[2:]
        ) != (horizon, nodes, channels):
            raise ValueError('reranker_weights axes do not match predictions')

        dtype = base_prediction.dtype
        correction = near_prediction - base_prediction
        retrieval_gap = retrieval_prediction - base_prediction
        history_distance = quality_retrieval[
            ..., channels:channels + 1
        ].to(dtype=dtype)
        future_dispersion = quality_retrieval[
            ..., channels + 1:channels + 2
        ].to(dtype=dtype)

        weights = reranker_weights.to(dtype=dtype).clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        candidate_count = weights.shape[1]
        if candidate_count > 1:
            entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=1)
            entropy = entropy / float(log(candidate_count))
            top_two = torch.topk(weights, k=2, dim=1).values
            margin = top_two[:, 0] - top_two[:, 1]
        else:
            entropy = torch.zeros_like(correction)
            margin = torch.ones_like(correction)

        features = torch.cat([
            correction,
            correction.abs(),
            retrieval_gap,
            retrieval_gap.abs(),
            torch.sign(history_distance) * torch.log1p(history_distance.abs()),
            torch.log1p(future_dispersion.abs()),
            entropy,
            margin,
        ], dim=-1)
        if not torch.isfinite(features).all():
            raise ValueError('correction utility gate inputs contain non-finite values')

        node_hidden = torch.matmul(
            self.node_factor, self.node_projection
        ).view(1, 1, self.num_nodes, self.hidden_dim)
        horizon_hidden = self.horizon_embedding.view(
            1, self.predict_length, 1, self.hidden_dim
        )
        hidden = F.gelu(
            self.feature_projection(features) + node_hidden + horizon_hidden
        )
        suppression = self.max_suppression * torch.tanh(
            self.output_projection(hidden)
        )
        gated_correction = (1.0 - suppression) * correction
        self.last_suppression = suppression.detach()
        self.last_correction = gated_correction.detach()
        return near_prediction + suppression * (
            base_prediction - near_prediction
        )


class AnalogAdaptiveResidualMixer(nn.Module):
    """ARM-style candidate residual mixer with a parent skip connection.

    Each train-memory candidate is represented by its residual to the current
    parent forecast and observable history features.  Attention is computed
    over candidates, and a bounded scalar controls the resulting residual
    delta.  The output projection is zero initialized, so the module is an
    identity at initialization and preserves the frozen parent forecast.
    """

    preserve_initialization = True

    def __init__(
        self,
        num_nodes,
        predict_length,
        candidate_feature_dim=8,
        hidden_dim=16,
        node_rank=2,
        max_scale=0.25,
    ):
        super().__init__()
        if num_nodes <= 0 or predict_length <= 0:
            raise ValueError('num_nodes and predict_length must be positive')
        if candidate_feature_dim <= 0 or hidden_dim <= 0 or node_rank <= 0:
            raise ValueError('mixer dimensions must be positive')
        if max_scale <= 0:
            raise ValueError('max_scale must be positive')
        self.num_nodes = int(num_nodes)
        self.predict_length = int(predict_length)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.node_rank = int(node_rank)
        self.max_scale = float(max_scale)

        # Candidate token = residual and its magnitude plus history features.
        self.candidate_projection = nn.Linear(
            2 + self.candidate_feature_dim, self.hidden_dim
        )
        # Query token = parent level and mean candidate residual.
        self.query_projection = nn.Linear(2, self.hidden_dim)
        self.horizon_embedding = nn.Parameter(torch.empty(
            self.predict_length, self.hidden_dim
        ))
        self.node_factor = nn.Parameter(torch.empty(
            self.num_nodes, self.node_rank
        ))
        self.node_projection = nn.Parameter(torch.empty(
            self.node_rank, self.hidden_dim
        ))
        self.score_projection = nn.Linear(self.hidden_dim, 1)
        self.output_projection = nn.Linear(2 * self.hidden_dim, 1)

        nn.init.xavier_uniform_(self.candidate_projection.weight)
        nn.init.zeros_(self.candidate_projection.bias)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.node_factor, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.node_projection)
        nn.init.xavier_uniform_(self.score_projection.weight)
        nn.init.zeros_(self.score_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.last_weights = None
        self.last_scale = None

    def forward(
        self,
        parent_prediction,
        candidate_future,
        candidate_available,
        candidate_features,
    ):
        if parent_prediction.ndim != 4:
            raise ValueError('parent_prediction must be shaped (B,H,N,C)')
        if candidate_future.ndim != 5 or candidate_available.ndim != 5:
            raise ValueError('candidate tensors must be shaped (B,K,H,N,C)')
        if candidate_future.shape != candidate_available.shape:
            raise ValueError('candidate future and availability must match')
        if candidate_features.ndim != 4:
            raise ValueError('candidate_features must be shaped (B,K,N,F)')
        batch, candidates, horizon, nodes, channels = candidate_future.shape
        if tuple(parent_prediction.shape) != (batch, horizon, nodes, channels):
            raise ValueError('candidate axes do not match parent prediction')
        if horizon != self.predict_length or nodes != self.num_nodes:
            raise ValueError('unexpected mixer horizon/node dimensions')
        if channels != 1:
            raise ValueError('adaptive residual mixer currently supports C=1')
        if tuple(candidate_features.shape) != (
            batch, candidates, nodes, self.candidate_feature_dim
        ):
            raise ValueError('candidate feature dimensions do not match mixer')

        dtype = parent_prediction.dtype
        candidates = candidate_future.to(dtype=dtype)
        available = (candidate_available.to(dtype=dtype) > 0).to(dtype=dtype)
        features = candidate_features.to(dtype=dtype)
        residual = candidates - parent_prediction[:, None]
        residual_mean = (
            (residual * available).sum(dim=1)
            / available.sum(dim=1).clamp_min(1.0)
        )
        candidate_input = torch.cat([
            residual,
            residual.abs(),
            features[:, :, None, :, :].expand(
                batch, candidates.shape[1], horizon, nodes,
                self.candidate_feature_dim
            ),
        ], dim=-1)
        candidate_hidden = self.candidate_projection(candidate_input)
        query_input = torch.cat([
            parent_prediction,
            residual_mean,
        ], dim=-1)
        query_hidden = self.query_projection(query_input)
        node_hidden = torch.matmul(
            self.node_factor, self.node_projection
        ).view(1, 1, self.num_nodes, self.hidden_dim)
        horizon_hidden = self.horizon_embedding.view(
            1, self.predict_length, 1, self.hidden_dim
        )
        candidate_hidden = F.gelu(
            candidate_hidden + node_hidden[:, None] + horizon_hidden[:, None]
        )
        query_hidden = F.gelu(query_hidden + node_hidden + horizon_hidden)
        logits = self.score_projection(candidate_hidden)
        masked_logits = torch.where(
            available > 0, logits, torch.full_like(logits, -1e9)
        )
        weights = torch.softmax(masked_logits, dim=1) * available
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        candidate_context = (weights * candidate_hidden).sum(dim=1)
        mixed_hidden = torch.cat([query_hidden, candidate_context], dim=-1)
        scale = self.max_scale * torch.tanh(
            self.output_projection(mixed_hidden)
        )
        mixed_residual = (weights * residual).sum(dim=1)
        self.last_weights = weights.detach()
        self.last_scale = scale.detach()
        return parent_prediction + scale * mixed_residual


class AnalogNodeHorizonRetrievalResidualAdapter(nn.Module):
    """Independent node/horizon analog blend with identity initialization.

    Unlike the rank-constrained adapter, this branch learns one bounded
    coefficient for every forecast-step/sensor pair.  The matrix is still
    tiny for PEMS04 (12 x 307 = 3,684 parameters), while avoiding the
    low-rank bottleneck observed in retrieval diagnostics.  Zero logits make
    epoch 0 exactly reproduce the frozen parent forecast.
    """

    preserve_initialization = True

    def __init__(
        self,
        num_nodes,
        predict_length,
        max_blend=1.0,
    ):
        super().__init__()
        if num_nodes <= 0 or predict_length <= 0:
            raise ValueError('num_nodes and predict_length must be positive')
        if not 0.0 < max_blend <= 1.0:
            raise ValueError(
                'analog_retrieval_max must be in the interval (0, 1]'
            )
        self.num_nodes = int(num_nodes)
        self.predict_length = int(predict_length)
        self.max_blend = float(max_blend)
        self.node_horizon_logits = nn.Parameter(torch.zeros(
            self.predict_length, self.num_nodes
        ))
        self.last_blend = None
        self.last_correction = None

    def forward(self, prediction, retrieval):
        if prediction.ndim != 4 or retrieval.ndim != 4:
            raise ValueError(
                'prediction and retrieval must be shaped (B,H,N,C)'
            )
        if prediction.shape != retrieval.shape:
            raise ValueError(
                'prediction and retrieval shapes must match exactly: '
                f'{tuple(prediction.shape)} != {tuple(retrieval.shape)}'
            )
        if prediction.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} forecast steps, got '
                f'{prediction.shape[1]}'
            )
        if prediction.shape[2] != self.num_nodes:
            raise ValueError(
                f'Expected {self.num_nodes} nodes, got '
                f'{prediction.shape[2]}'
            )
        if not torch.isfinite(retrieval).all():
            raise ValueError('retrieval contains non-finite values')

        blend = self.max_blend * torch.tanh(
            self.node_horizon_logits
        ).view(1, self.predict_length, self.num_nodes, 1)
        correction = blend * (retrieval - prediction)
        self.last_blend = blend.detach()
        self.last_correction = correction.detach()
        return prediction + correction


class AnalogConfidenceAwareRetrievalResidualAdapter(nn.Module):
    """Local confidence refinement for a frozen rank-2 + exact parent.

    The parent rank-2 gate is fixed for every query at a given node/horizon.
    This stage learns only a small *additional* blend coefficient from
    retrieval quality at each (sample, horizon, node):

      - normalized KNN history distance,
      - normalized dispersion of the retrieved future continuations,
      - whether an exact-history continuation was available.

    All coefficients are zero-initialized, so construction is a strict
    identity and epoch-0 predictions exactly match the frozen parent.  The
    bounded adjustment prevents this 48-parameter refinement from replacing
    the already validated parent retrieval gate.
    """

    preserve_initialization = True

    def __init__(
        self,
        predict_length,
        max_adjustment=0.25,
        num_nodes=None,
        node_rank=0,
        feature_dim=3,
    ):
        super().__init__()
        if predict_length <= 0:
            raise ValueError('predict_length must be positive')
        if not 0.0 < max_adjustment <= 1.0:
            raise ValueError(
                'analog_confidence_max must be in the interval (0, 1]'
            )
        self.predict_length = int(predict_length)
        self.max_adjustment = float(max_adjustment)
        self.num_nodes = int(num_nodes or 0)
        self.node_rank = int(node_rank)
        self.feature_dim = int(feature_dim)
        if self.feature_dim not in (3, 6):
            raise ValueError(
                'analog_confidence_feature_dim must be 3 (legacy) or 6'
            )
        self.quality_weight = nn.Parameter(torch.zeros(
            self.predict_length, self.feature_dim
        ))
        self.horizon_bias = nn.Parameter(torch.zeros(self.predict_length))
        self.node_aware = self.num_nodes > 0 and self.node_rank > 0
        if self.node_aware:
            # Low-rank node-conditioned correction to the shared horizon gate.
            # Parameter count is N*r + H*3*r instead of N*H*3.
            self.node_factor = nn.Parameter(torch.zeros(self.num_nodes, self.node_rank))
            self.node_quality_factor = nn.Parameter(
                torch.zeros(
                    self.predict_length, self.feature_dim, self.node_rank
                )
            )

        self.last_adjustment = None
        self.last_correction = None

    def forward(self, prediction, retrieval, exact_available=None):
        if prediction.ndim != 4 or retrieval.ndim != 4:
            raise ValueError(
                'prediction and retrieval must be shaped (B,H,N,C)'
            )
        if prediction.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} horizons, got '
                f'{prediction.shape[1]}'
            )
        if retrieval.shape[:-1] != prediction.shape[:-1]:
            raise ValueError(
                'Retrieval and prediction batch/horizon/node shapes must '
                f'match: {tuple(retrieval.shape)} vs '
                f'{tuple(prediction.shape)}'
            )
        forecast_channels = prediction.shape[-1]
        if retrieval.shape[-1] != forecast_channels + 2:
            raise ValueError(
                'Confidence-aware refinement expects analog forecast channels '
                'followed by history_distance and future_dispersion; '
                f'got shape {tuple(retrieval.shape)}'
            )
        if not torch.isfinite(retrieval).all():
            raise ValueError('retrieval contains non-finite values')

        analog_forecast = retrieval[..., :forecast_channels]
        history_distance = retrieval[..., forecast_channels]
        future_dispersion = retrieval[..., forecast_channels + 1]

        # The loader already converts both quantities to z-score units.
        # log1p followed by x/(1+x) keeps extreme retrieval failures bounded
        # while preserving the reliability ordering.
        history_distance = torch.log1p(
            torch.clamp(history_distance, min=0.0)
        )
        future_dispersion = torch.log1p(
            torch.clamp(future_dispersion, min=0.0)
        )
        history_distance = history_distance / (1.0 + history_distance)
        future_dispersion = future_dispersion / (1.0 + future_dispersion)

        exact_feature = torch.zeros_like(history_distance)
        if exact_available is not None:
            if exact_available.shape != prediction.shape:
                raise ValueError(
                    'exact_available must match prediction shape; '
                    f'{tuple(exact_available.shape)} != {tuple(prediction.shape)}'
                )
            exact_feature = exact_available.float().mean(dim=-1)

        quality_features = [
            history_distance, future_dispersion, exact_feature
        ]
        if self.feature_dim == 6:
            # Query-local risk signals.  Unlike the target, both forecasts are
            # available at inference time.  Bound every feature to keep the
            # relative gap stable when the parent prediction is near zero.
            signed_gap = (analog_forecast - prediction).mean(dim=-1)
            absolute_gap = (analog_forecast - prediction).abs().mean(dim=-1)
            relative_gap = (
                (analog_forecast - prediction)
                / prediction.abs().clamp_min(1e-3)
            ).mean(dim=-1)
            signed_gap = signed_gap / (1.0 + signed_gap.abs())
            absolute_gap = absolute_gap / (1.0 + absolute_gap)
            relative_gap = relative_gap / (1.0 + relative_gap.abs())
            quality_features.extend([
                signed_gap, absolute_gap, relative_gap
            ])
        quality = torch.stack(quality_features, dim=-1)
        adjustment_logits = torch.einsum(
            'bhni,hi->bhn', quality, self.quality_weight
        )
        if self.node_aware:
            node_delta = torch.einsum(
                'bhni,hir,nr->bhn', quality, self.node_quality_factor,
                self.node_factor
            )
            adjustment_logits = adjustment_logits + node_delta
        adjustment_logits = (
            adjustment_logits
            + self.horizon_bias.view(1, self.predict_length, 1)
        )
        adjustment = self.max_adjustment * torch.tanh(
            adjustment_logits
        )
        adjustment = adjustment.unsqueeze(-1)
        correction = adjustment * (analog_forecast - prediction)

        self.last_adjustment = adjustment.detach()
        self.last_correction = correction.detach()
        return prediction + correction


class AnalogExactMatchResidualAdapter(nn.Module):
    """Trust train-memory continuations only for exact observed histories.

    ``exact_retrieval`` and ``available`` are precomputed from fixed
    training-split candidates at the same time of day.  Training queries use
    leave-one-out matching.  One zero-initialized bounded coefficient per
    horizon makes the complete model exactly equal to its frozen parent at
    epoch 0 and lets ordinary training decide how strongly to trust exact
    continuations.
    """

    preserve_initialization = True

    def __init__(self, predict_length, max_blend=1.0):
        super().__init__()
        if predict_length <= 0:
            raise ValueError('predict_length must be positive')
        if not 0.0 < max_blend <= 1.0:
            raise ValueError(
                'analog_exact_match_max must be in the interval (0, 1]'
            )
        self.predict_length = int(predict_length)
        self.max_blend = float(max_blend)
        self.horizon_logits = nn.Parameter(torch.zeros(self.predict_length))
        self.last_horizon_blend = None
        self.last_selected_fraction = None
        self.last_correction = None

    def forward(self, prediction, exact_retrieval, available):
        if not (
            prediction.ndim == exact_retrieval.ndim == available.ndim == 4
        ):
            raise ValueError(
                'Exact-match tensors must be shaped (B,H,N,C)'
            )
        if not (
            prediction.shape == exact_retrieval.shape == available.shape
        ):
            raise ValueError('Exact-match tensor shapes must match exactly')
        if prediction.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} forecast steps, got '
                f'{prediction.shape[1]}'
            )
        if not (
            torch.isfinite(exact_retrieval).all()
            and torch.isfinite(available).all()
        ):
            raise ValueError('Exact-match inputs contain non-finite values')
        if not torch.all((available == 0) | (available == 1)):
            raise ValueError('Exact-match availability must be binary')

        horizon_blend = self.max_blend * torch.tanh(
            self.horizon_logits
        ).view(1, self.predict_length, 1, 1)
        correction = (
            horizon_blend
            * available.to(dtype=prediction.dtype)
            * (exact_retrieval - prediction)
        )
        self.last_horizon_blend = horizon_blend.detach()
        self.last_selected_fraction = available.float().mean().detach()
        self.last_correction = correction.detach()
        return prediction + correction


class NodeLinearDynamicsBypass(nn.Module):
    """Small local linear bypass for node-specific short-term dynamics.

    The branch uses only the observable history.  For every node and forecast
    horizon it linearly maps (1) the trajectory relative to the latest value
    and (2) mean absolute adjacent variation to an additive forecast-delta
    correction.  There is deliberately no bias: static node/horizon offsets
    belong to the main predictor and a bias here would behave like calibration.

    The weight tensor is exactly zero at construction.  Together with
    ``preserve_initialization`` this guarantees that enabling the branch leaves
    the initial forecast bit-identical to the parent model while still giving
    the weights a non-zero gradient on the first optimization step.
    """

    preserve_initialization = True

    def __init__(
        self,
        num_nodes,
        input_length,
        predict_length,
        pre_dim=1,
        scale=0.25,
    ):
        super().__init__()
        if num_nodes <= 0:
            raise ValueError('num_nodes must be positive')
        if input_length < 2:
            raise ValueError(
                'NodeLinearDynamicsBypass requires at least 2 history steps'
            )
        if predict_length <= 0 or pre_dim <= 0:
            raise ValueError('predict_length and pre_dim must be positive')
        if not 0.0 < scale <= 1.0:
            raise ValueError(
                'node_linear_dynamics_scale must be in the interval (0, 1]'
            )

        self.num_nodes = int(num_nodes)
        self.input_length = int(input_length)
        self.predict_length = int(predict_length)
        self.pre_dim = int(pre_dim)
        self.scale = float(scale)
        # T-1 non-trivial relative-history values plus one mean-variation
        # feature.  The latest-minus-latest value is identically zero and is
        # deliberately omitted, leaving exactly T effective features.
        self.num_features = self.input_length
        self.weight = nn.Parameter(torch.zeros(
            self.num_nodes,
            self.pre_dim,
            self.predict_length,
            self.num_features,
        ))
        self.last_correction = None

    def forward(self, history):
        # history: (B, T, N, pre_dim), in the same normalized space as the
        # main predictor and latest-value residual.
        if history.ndim != 4:
            raise ValueError(
                'NodeLinearDynamicsBypass expects history shaped (B,T,N,C)'
            )
        if history.shape[1] != self.input_length:
            raise ValueError(
                f'Expected {self.input_length} history steps, got '
                f'{history.shape[1]}'
            )
        if history.shape[2] != self.num_nodes:
            raise ValueError(
                f'Expected {self.num_nodes} nodes, got {history.shape[2]}'
            )
        if history.shape[3] != self.pre_dim:
            raise ValueError(
                f'Expected {self.pre_dim} target channels, got '
                f'{history.shape[3]}'
            )

        latest = history[:, -1:, :, :]
        relative_trajectory = history[:, :-1, :, :] - latest
        volatility = (
            history[:, 1:, :, :] - history[:, :-1, :, :]
        ).abs().mean(dim=1, keepdim=True)
        features = torch.cat(
            [relative_trajectory, volatility], dim=1
        )  # (B, T, N, C): T-1 relative values + 1 volatility

        correction = torch.einsum(
            'bfnc,nchf->bhnc', features, self.weight
        )
        correction = self.scale * correction
        self.last_correction = correction.detach()
        return correction


class RegimeGatedResidualAdapter(nn.Module):
    """Shared nonlinear residual adapter for observable change regimes.

    The frozen TLAST parent already predicts the dominant traffic level.  This
    adapter only learns the remaining correction from information available at
    inference time: the 12-step relative trajectory, signed first and second
    differences, summary change statistics, the parent's forecast delta, and
    a horizon embedding.  Parameters are shared across nodes so the adapter
    learns transferable rise/fall/reversal patterns instead of memorizing a
    separate correction for every sensor.

    The final projection is exactly zero initialized.  Enabling the adapter is
    therefore bit-identical to the parent at epoch 0, while a standard MAE
    gradient immediately updates the final projection.  There is no target
    leakage, validation calibration, loss reweighting, or persistence-delta
    scaling in this module.
    """

    preserve_initialization = True

    def __init__(
        self,
        input_length,
        predict_length,
        pre_dim=1,
        hidden_dim=32,
        horizon_dim=8,
        scale=0.25,
    ):
        super().__init__()
        if input_length < 3:
            raise ValueError(
                'RegimeGatedResidualAdapter requires at least 3 history steps'
            )
        if predict_length <= 0 or pre_dim <= 0:
            raise ValueError('predict_length and pre_dim must be positive')
        if hidden_dim <= 0 or horizon_dim <= 0:
            raise ValueError('hidden_dim and horizon_dim must be positive')
        if not 0.0 < scale <= 1.0:
            raise ValueError(
                'regime_residual_scale must be in the interval (0, 1]'
            )

        self.input_length = int(input_length)
        self.predict_length = int(predict_length)
        self.pre_dim = int(pre_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon_dim = int(horizon_dim)
        self.scale = float(scale)

        # (T-1) relative values + (T-1) first differences + (T-2)
        # second differences + volatility, signed trend and curvature.
        self.num_history_features = 3 * self.input_length - 1
        self.history_encoder = nn.Sequential(
            nn.Linear(self.num_history_features, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.horizon_embedding = nn.Embedding(
            self.predict_length, self.horizon_dim
        )
        joint_dim = self.hidden_dim + self.horizon_dim + 1
        gate_hidden = max(8, self.hidden_dim // 2)
        self.regime_gate = nn.Sequential(
            nn.Linear(joint_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.correction_net = nn.Sequential(
            nn.Linear(joint_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.correction_net[-1].weight)
        nn.init.zeros_(self.correction_net[-1].bias)
        self.last_gate = None
        self.last_correction = None

    def _history_features(self, history):
        if history.ndim != 4:
            raise ValueError(
                'RegimeGatedResidualAdapter expects history shaped (B,T,N,C)'
            )
        if history.shape[1] != self.input_length:
            raise ValueError(
                f'Expected {self.input_length} history steps, got '
                f'{history.shape[1]}'
            )
        if history.shape[3] != self.pre_dim:
            raise ValueError(
                f'Expected {self.pre_dim} target channels, got '
                f'{history.shape[3]}'
            )

        target = history[..., :self.pre_dim]
        latest = target[:, -1:, :, :]
        relative = target[:, :-1, :, :] - latest
        first_difference = target[:, 1:, :, :] - target[:, :-1, :, :]
        second_difference = (
            first_difference[:, 1:, :, :]
            - first_difference[:, :-1, :, :]
        )
        volatility = first_difference.abs().mean(dim=1, keepdim=True)
        signed_trend = latest - target[:, :1, :, :]
        curvature = (
            first_difference[:, -1:, :, :]
            - first_difference[:, :1, :, :]
        )
        features = torch.cat(
            [
                relative,
                first_difference,
                second_difference,
                volatility,
                signed_trend,
                curvature,
            ],
            dim=1,
        )
        # (B, F, N, C) -> (B, N, C, F)
        return features.permute(0, 2, 3, 1).contiguous()

    def forward(self, history, parent_delta):
        if parent_delta.ndim != 4:
            raise ValueError(
                'parent_delta must be shaped (B,H,N,C)'
            )
        if parent_delta.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} forecast steps, got '
                f'{parent_delta.shape[1]}'
            )
        if parent_delta.shape[3] != self.pre_dim:
            raise ValueError(
                f'Expected parent_delta pre_dim={self.pre_dim}, got '
                f'{parent_delta.shape[3]}'
            )
        if (
            parent_delta.shape[0] != history.shape[0]
            or parent_delta.shape[2] != history.shape[2]
        ):
            raise ValueError(
                'history and parent_delta must have matching batch and node '
                'dimensions'
            )

        history_features = self._history_features(history)
        context = self.history_encoder(history_features)  # (B,N,C,D)
        batch_size, num_nodes, channels, _ = context.shape
        context = context.unsqueeze(1).expand(
            -1, self.predict_length, -1, -1, -1
        )
        horizon = self.horizon_embedding.weight.view(
            1, self.predict_length, 1, 1, self.horizon_dim
        ).expand(batch_size, -1, num_nodes, channels, -1)
        joint = torch.cat(
            [context, horizon, parent_delta.unsqueeze(-1)], dim=-1
        )
        gate = torch.sigmoid(self.regime_gate(joint))
        raw_correction = self.correction_net(joint)
        correction = self.scale * gate * raw_correction
        correction = correction.squeeze(-1)  # (B,H,N,C)
        self.last_gate = gate.squeeze(-1).detach()
        self.last_correction = correction.detach()
        return correction


class MultiResolutionBlockResidualAdapter(nn.Module):
    """Coarse-to-fine residual correction over fixed future blocks.

    The parent retains its full 12-step forecast.  This adapter predicts only
    four low-frequency corrections (one for each consecutive 3-step block)
    from the observable relative history, signed changes, and the parent's
    forecast-delta trajectory.  Repeating each correction within its block
    cannot replace the parent's fine temporal shape; it can only correct the
    slowly accumulating level drift targeted by the auxiliary block-mean MAE.

    The block projection is zero initialized so epoch 0 is bit-identical to
    the frozen parent.
    """

    preserve_initialization = True

    def __init__(
        self,
        input_length,
        predict_length,
        pre_dim=1,
        block_size=3,
        hidden_dim=32,
        scale=0.25,
    ):
        super().__init__()
        if input_length < 2:
            raise ValueError(
                'MultiResolutionBlockResidualAdapter requires at least 2 '
                'history steps'
            )
        if predict_length <= 0 or pre_dim <= 0:
            raise ValueError('predict_length and pre_dim must be positive')
        if block_size <= 0 or predict_length % block_size != 0:
            raise ValueError(
                'predict_length must be divisible by multires_block_size'
            )
        if hidden_dim <= 0:
            raise ValueError('multires_block_hidden must be positive')
        if not 0.0 < scale <= 1.0:
            raise ValueError(
                'multires_block_scale must be in the interval (0, 1]'
            )

        self.input_length = int(input_length)
        self.predict_length = int(predict_length)
        self.pre_dim = int(pre_dim)
        self.block_size = int(block_size)
        self.num_blocks = self.predict_length // self.block_size
        self.hidden_dim = int(hidden_dim)
        self.scale = float(scale)
        self.num_features = (
            2 * (self.input_length - 1) + self.predict_length
        )

        self.feature_encoder = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.block_projection = nn.Linear(
            self.hidden_dim, self.num_blocks
        )
        nn.init.zeros_(self.block_projection.weight)
        nn.init.zeros_(self.block_projection.bias)
        self.last_block_correction = None
        self.last_correction = None

    def forward(self, history, parent_delta):
        if history.ndim != 4 or parent_delta.ndim != 4:
            raise ValueError(
                'history and parent_delta must both be four-dimensional'
            )
        if history.shape[1] != self.input_length:
            raise ValueError(
                f'Expected {self.input_length} history steps, got '
                f'{history.shape[1]}'
            )
        if parent_delta.shape[1] != self.predict_length:
            raise ValueError(
                f'Expected {self.predict_length} forecast steps, got '
                f'{parent_delta.shape[1]}'
            )
        if history.shape[3] != self.pre_dim:
            raise ValueError(
                f'Expected history pre_dim={self.pre_dim}, got '
                f'{history.shape[3]}'
            )
        if parent_delta.shape[3] != self.pre_dim:
            raise ValueError(
                f'Expected parent_delta pre_dim={self.pre_dim}, got '
                f'{parent_delta.shape[3]}'
            )
        if (
            parent_delta.shape[0] != history.shape[0]
            or parent_delta.shape[2] != history.shape[2]
        ):
            raise ValueError(
                'history and parent_delta must have matching batch and node '
                'dimensions'
            )

        latest = history[:, -1:, :, :]
        relative = history[:, :-1, :, :] - latest
        first_difference = history[:, 1:, :, :] - history[:, :-1, :, :]
        features = torch.cat(
            [relative, first_difference, parent_delta], dim=1
        )
        features = features.permute(0, 2, 3, 1).contiguous()
        context = self.feature_encoder(features)
        block_correction = self.scale * self.block_projection(context)
        # (B,N,C,K) -> (B,K,N,C) -> (B,H,N,C)
        block_correction = block_correction.permute(0, 3, 1, 2)
        correction = block_correction.repeat_interleave(
            self.block_size, dim=1
        )
        self.last_block_correction = block_correction.detach()
        self.last_correction = correction.detach()
        return correction


class StudentRiskHead(nn.Module):
    """Tiny sample-risk estimator using inference-observable summaries only."""

    def __init__(self, slice_size_per_day, hidden_dim=16):
        super().__init__()
        self.slice_size_per_day = int(slice_size_per_day)
        self.network = nn.Sequential(
            nn.Linear(16, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    @staticmethod
    def _summaries(value):
        flat = value.reshape(value.shape[0], -1)
        temporal_change = (value[:, 1:] - value[:, :-1]).abs()
        return [
            flat.mean(dim=1),
            flat.std(dim=1, unbiased=False),
            flat.abs().mean(dim=1),
            flat.amin(dim=1),
            flat.amax(dim=1),
            temporal_change.reshape(value.shape[0], -1).mean(dim=1),
        ]

    def forward(self, history, prediction, x_time):
        history = history.detach()
        prediction = prediction.detach()
        features = self._summaries(history) + self._summaries(prediction)
        last_history = history[:, -1].reshape(history.shape[0], -1)
        first_prediction = prediction[:, 0].reshape(prediction.shape[0], -1)
        features.extend([
            last_history.mean(dim=1),
            last_history.std(dim=1, unbiased=False),
        ])
        if x_time is None:
            tod_sin = history.new_zeros(history.shape[0])
            tod_cos = history.new_zeros(history.shape[0])
        else:
            tod = x_time[:, -1, 0].to(history.dtype)
            angle = 2.0 * torch.pi * tod / float(self.slice_size_per_day)
            tod_sin, tod_cos = torch.sin(angle), torch.cos(angle)
        features.extend([tod_sin, tod_cos])
        design = torch.stack(features, dim=-1)
        if design.shape[-1] != 16:
            raise RuntimeError(f'Risk-head feature count is {design.shape[-1]}, expected 16')
        return self.network(design).squeeze(-1)


class SparseNeighborInnovation(nn.Module):
    """Observed, Train-selected neighbor changes injected into the time trunk."""

    def __init__(self, neighbor_indices, channels):
        super().__init__()
        neighbors = torch.as_tensor(neighbor_indices, dtype=torch.long)
        if neighbors.ndim != 2:
            raise ValueError('neighbor_indices must have shape (nodes, k)')
        self.register_buffer('neighbors', neighbors)
        self.temporal = nn.Conv1d(1, channels, kernel_size=3, padding=1, bias=False)
        self.gate = nn.Parameter(torch.zeros(channels))
        # run.py normally Xavier-reinitializes every matrix, including buffers'
        # surrounding modules. Preserve the exact zero-output starting point.
        self.preserve_initialization = True

    def forward(self, history):
        own = history[..., 0]
        neighbor_mean = own[:, :, self.neighbors].mean(dim=-1)
        difference = neighbor_mean - own
        innovation = torch.cat((
            torch.zeros_like(difference[:, :1]),
            difference[:, 1:] - difference[:, :-1],
        ), dim=1)
        batch, steps, nodes = innovation.shape
        temporal = self.temporal(
            innovation.permute(0, 2, 1).reshape(batch * nodes, 1, steps)
        )
        temporal = temporal.reshape(batch, nodes, -1, steps).permute(0, 3, 1, 2)
        return torch.tanh(self.gate)[None, None, None, :] * temporal


class Model(nn.Module):
    """
    TLAST 模型的主类（改进版）
    
    改进点：
        1. 使用 MS_TCN（多尺度时间卷积）替代原始的单尺度 TCN
        2. 使用 FTLE（全时滞嵌入）替代原始的单参考时滞嵌入
        3. 使用 MAPA_EncoderLayer（记忆增强原型注意力）替代原始的 EncoderLayer
        4. 支持正交损失（orthogonal loss）以防止 Memory Bank 原型坍缩
    """
    def __init__(self, args):
        """
        模型初始化
        
        参数：
            args: 包含所有模型超参数的命名空间
        """
        super().__init__()
        
        # ===== 基础参数 =====
        self.input_length = args.input_length
        self.predict_length = args.predict_length
        self.in_dim = args.in_dim
        self.pre_dim = args.pre_dim
        self.num_nodes = args.num_nodes
        self.tau = args.tau
        self.useTCN = self.tau > 0
        # stack_blocks 必须在 useTCN 判断之外初始化：useTCN=False（tau=0）时
        # forward 仍会读取 self.stack_blocks 来构造 _stack_mode，否则触发 AttributeError。
        self.stack_blocks = bool(getattr(args, 'stack_blocks', False))
        self.hid_dim = args.hid_dim
        self.n_heads = args.n_heads
        self.M = args.M
        self.d_out = args.d_out
        self.query_cross_time = bool(args.query_cross_time)
        self.return_att = bool(args.return_att)
        self.num_layers = args.num_layers
        self.norm_flag = args.norm_flag
        self.att_type = args.att_type
        self.activation_data = args.activation_data
        self.activation_enc = args.activation_enc
        self.activation_dec = args.activation_dec
        self.emb_dropout = args.emb_dropout
        self.te_emb_dropout = args.te_emb_dropout
        self.se_emb_dropout = args.se_emb_dropout
        self.tle_emb_dropout = args.tle_emb_dropout
        self.enc_dropout = args.enc_dropout
        self.att_dropout = args.att_dropout
        self.prob_factor = args.prob_factor
        self.slice_size_per_day = args.slice_size_per_day
        self.use_student_risk_head = bool(getattr(
            args, 'use_student_risk_head', 0
        ))
        self.student_risk_hidden = int(getattr(
            args, 'student_risk_hidden', 16
        ))
        self.last_risk_score = None
        self.use_elastic_rank_router = bool(getattr(
            args, 'use_elastic_rank_router', 0
        ))
        self.elastic_rank_temperature = float(getattr(
            args, 'elastic_rank_temperature', 1.0
        ))
        if self.use_elastic_rank_router:
            self.elastic_rank_router = ElasticRankRouter(
                hidden=int(getattr(args, 'elastic_router_hidden', 8)),
                initial_high_probability=float(getattr(
                    args, 'elastic_initial_high_probability', 0.8
                )),
            )
        
        # Graph convolution parameters (保留原始功能)
        self.use_graph_conv = bool(getattr(args, 'use_graph_conv', 0))
        self.graph_conv_type = getattr(args, 'graph_conv_type', 'adaptive')
        self.k_hops = getattr(args, 'k_hops', [1, 2])
        
        # FTLE 和 MAPA 的开关参数
        self.use_ftle = bool(getattr(args, 'use_ftle', True))
        self.use_time_lag_embedding = bool(
            getattr(args, 'use_time_lag_embedding', 1)
        )
        self.ftle_mode = getattr(args, 'ftle_mode', 'full')
        if self.ftle_mode not in ('full', 'gated_blend'):
            raise ValueError(f'Unsupported ftle_mode: {self.ftle_mode}')
        self.single_tle_mode = getattr(args, 'single_tle_mode', 'last')
        if self.single_tle_mode not in ('last', 'selective'):
            raise ValueError(
                f'Unsupported single_tle_mode: {self.single_tle_mode}'
            )
        if self.use_ftle and self.single_tle_mode != 'last':
            raise ValueError(
                'single_tle_mode="selective" requires use_ftle=0'
            )
        if not self.use_time_lag_embedding and (
            self.use_ftle or self.single_tle_mode != 'last'
        ):
            raise ValueError(
                'use_time_lag_embedding=0 requires use_ftle=0 and '
                'single_tle_mode="last"'
            )
        self.use_mapa = bool(getattr(args, 'use_mapa', True))
        self.lambda_ortho = getattr(args, 'lambda_ortho', 0.01)
        self.num_prototypes = getattr(args, 'num_prototypes', 16)
        # 空间聚合层类型开关：'mapa'（默认，记忆增强原型注意力）或 'agcrn'（AGCRN 式图卷积替换 MAPA）
        self.spatial_type = getattr(args, 'spatial_type', 'mapa')
        self.original_spa_dynamic_proxy = bool(
            getattr(args, 'original_spa_dynamic_proxy', True)
        )
        if self.spatial_type == 'original_spa':
            # EncoderLayer follows the original TLAST proxy-SPA return contract,
            # not MAPA's (output, orthogonal_loss) contract.
            self.use_mapa = False
        self.use_future_time_decoder = bool(
            getattr(args, 'use_future_time_decoder', 0)
        )
        self.use_volatility_delta_scale = bool(
            getattr(args, 'use_volatility_delta_scale', 0)
        )
        self.volatility_delta_hidden = int(
            getattr(args, 'volatility_delta_hidden', 8)
        )
        self.volatility_delta_max = float(
            getattr(args, 'volatility_delta_max', 0.1)
        )
        self.use_node_linear_dynamics = bool(
            getattr(args, 'use_node_linear_dynamics', 0)
        )
        self.node_linear_dynamics_scale = float(
            getattr(args, 'node_linear_dynamics_scale', 0.25)
        )
        self.use_regime_gated_residual = bool(
            getattr(args, 'use_regime_gated_residual', 0)
        )
        self.regime_residual_hidden = int(
            getattr(args, 'regime_residual_hidden', 32)
        )
        self.regime_residual_horizon_dim = int(
            getattr(args, 'regime_residual_horizon_dim', 8)
        )
        self.regime_residual_scale = float(
            getattr(args, 'regime_residual_scale', 0.25)
        )
        self.use_multires_block_residual = bool(
            getattr(args, 'use_multires_block_residual', 0)
        )
        self.multires_block_size = int(
            getattr(args, 'multires_block_size', 3)
        )
        self.multires_block_hidden = int(
            getattr(args, 'multires_block_hidden', 32)
        )
        self.multires_block_scale = float(
            getattr(args, 'multires_block_scale', 0.25)
        )
        # ===== 零初始化季节漂移残差（TOD 漂移，冻结父模型上训练）=====
        # 核心输出：y_hat = y_parent + tanh(a_h) * (mu_TOD_future - mu_TOD_last)
        # a_h 为 predict_length 个可学习标量，初始化为 0 -> tanh(0)=0 -> 初始输出
        # 与父模型逐元素一致（干净对照）。仅用训练集 TOD×传感器均值 base_mean，
        # 不读未来真实流量，是模型前向的一部分（非 test calibration）。
        self.use_seasonal_drift_residual = bool(
            getattr(args, 'use_seasonal_drift_residual', 0)
        )
        self.seasonal_drift_max = float(
            getattr(args, 'seasonal_drift_max', 1.0)
        )
        self.use_analog_retrieval_residual = bool(
            getattr(args, 'use_analog_retrieval_residual', 0)
        )
        self.analog_retrieval_max = float(
            getattr(args, 'analog_retrieval_max', 1.0)
        )
        self.use_analog_lowrank_retrieval_residual = bool(
            getattr(args, 'use_analog_lowrank_retrieval_residual', 0)
        )
        self.use_analog_temporal_lowrank_decoder = bool(
            getattr(args, 'use_analog_temporal_lowrank_decoder', 0)
        )
        self.analog_temporal_decoder_rank = int(
            getattr(args, 'analog_temporal_decoder_rank', 4)
        )
        self.analog_temporal_decoder_max_scale = float(
            getattr(args, 'analog_temporal_decoder_max_scale', 0.25)
        )
        self.use_analog_temporal_gap_residual = bool(
            getattr(args, 'use_analog_temporal_gap_residual', 0)
        )
        self.analog_temporal_gap_rank = int(
            getattr(args, 'analog_temporal_gap_rank', 4)
        )
        self.analog_temporal_gap_max = float(
            getattr(args, 'analog_temporal_gap_max', 0.25)
        )
        self.use_analog_node_horizon_retrieval_residual = bool(
            getattr(
                args,
                'use_analog_node_horizon_retrieval_residual',
                0,
            )
        )
        self.use_analog_confidence_aware_retrieval_residual = bool(
            getattr(args, 'use_analog_confidence_aware_retrieval_residual', 0)
        )
        self.use_analog_weekly_retrieval_residual = bool(
            getattr(args, 'use_analog_weekly_retrieval_residual', 0)
        )
        self.use_analog_metric_retrieval_residual = bool(
            getattr(args, 'use_analog_metric_retrieval_residual', 0)
        )
        self.analog_confidence_max = float(
            getattr(args, 'analog_confidence_max', 0.25)
        )
        self.analog_confidence_node_rank = int(
            getattr(args, 'analog_confidence_node_rank', 0)
        )
        self.analog_confidence_feature_dim = int(
            getattr(args, 'analog_confidence_feature_dim', 3)
        )
        self.analog_weekly_retrieval_max = float(
            getattr(args, 'analog_weekly_retrieval_max', 0.5)
        )
        self.analog_weekly_rank = int(
            getattr(args, 'analog_weekly_rank', 2)
        )
        self.analog_metric_retrieval_max = float(
            getattr(args, 'analog_metric_retrieval_max', 0.5)
        )
        self.analog_metric_rank = int(
            getattr(args, 'analog_metric_rank', 2)
        )
        self.analog_lowrank_rank = int(
            getattr(args, 'analog_lowrank_rank', 2)
        )
        if (
            self.use_analog_confidence_aware_retrieval_residual
            and not self.use_analog_lowrank_retrieval_residual
        ):
            raise ValueError(
                'confidence-aware retrieval is a refinement of the validated '
                'rank-2 parent and requires '
                'use_analog_lowrank_retrieval_residual=1'
            )
        if (
            self.use_analog_weekly_retrieval_residual
            and not self.use_analog_lowrank_retrieval_residual
        ):
            raise ValueError(
                'weekly retrieval is a refinement of the validated daily '
                'rank-2 parent and requires '
                'use_analog_lowrank_retrieval_residual=1'
            )
        if (
            self.use_analog_metric_retrieval_residual
            and not self.use_analog_lowrank_retrieval_residual
        ):
            raise ValueError(
                'future-consistent metric retrieval is a refinement of the '
                'validated daily rank-2 parent and requires '
                'use_analog_lowrank_retrieval_residual=1'
            )
        if (
            self.use_analog_weekly_retrieval_residual
            and self.use_analog_confidence_aware_retrieval_residual
        ):
            raise ValueError(
                'weekly and confidence-aware retrieval are mutually '
                'exclusive single-variable refinements'
            )
        if (
            self.use_analog_metric_retrieval_residual
            and (
                self.use_analog_weekly_retrieval_residual
                or self.use_analog_confidence_aware_retrieval_residual
            )
        ):
            raise ValueError(
                'future-consistent metric, weekly, and confidence-aware '
                'retrieval are mutually exclusive refinements'
            )
        self.use_analog_exact_match_residual = bool(
            getattr(args, 'use_analog_exact_match_residual', 0)
        )
        self.analog_exact_match_max = float(
            getattr(args, 'analog_exact_match_max', 1.0)
        )
        self.use_analog_candidate_reranker = bool(
            getattr(args, 'use_analog_candidate_reranker', 0)
        )
        self.use_analog_direct_prediction = bool(
            getattr(args, 'use_analog_direct_prediction', 0)
        )
        self.analog_candidate_feature_dim = int(
            getattr(args, 'analog_candidate_feature_dim', 8)
        )
        self.analog_candidate_disagreement_dim = int(
            getattr(args, 'analog_candidate_disagreement_dim', 6)
        )
        self.analog_candidate_parent_residual_features = bool(
            getattr(args, 'analog_candidate_parent_residual_features', 0)
        )
        self.analog_candidate_hidden = int(
            getattr(args, 'analog_candidate_hidden', 16)
        )
        self.use_analog_query_conditioned_residual = bool(
            getattr(args, 'use_analog_query_conditioned_residual', 0)
        )
        self.analog_query_conditioned_hidden = int(
            getattr(args, 'analog_query_conditioned_hidden', 8)
        )
        self.analog_query_conditioned_max = float(
            getattr(args, 'analog_query_conditioned_max', 0.25)
        )
        self.use_analog_code_domain_aggregation = bool(
            getattr(args, 'use_analog_code_domain_aggregation', 0)
        )
        self.analog_candidate_code_rank = int(
            getattr(args, 'analog_candidate_code_rank', 0)
        )
        self.analog_candidate_code_mean = getattr(
            args, 'analog_candidate_code_mean', None
        )
        self.analog_candidate_code_basis = getattr(
            args, 'analog_candidate_code_basis', None
        )
        self.analog_code_domain_decode_candidates = bool(
            getattr(args, 'analog_code_domain_decode_candidates', 1)
        )
        if self.use_analog_code_domain_aggregation and not self.use_analog_candidate_reranker:
            raise ValueError(
                'Code-domain aggregation requires use_analog_candidate_reranker=1'
            )
        if self.use_analog_query_conditioned_residual and not self.use_analog_candidate_reranker:
            raise ValueError(
                'Query-conditioned residual requires use_analog_candidate_reranker=1'
            )
        self.use_analog_correction_utility_gate = bool(
            getattr(args, 'use_analog_correction_utility_gate', 0)
        )
        self.analog_correction_utility_hidden = int(
            getattr(args, 'analog_correction_utility_hidden', 8)
        )
        self.analog_correction_utility_node_rank = int(
            getattr(args, 'analog_correction_utility_node_rank', 2)
        )
        self.analog_correction_utility_max = float(
            getattr(args, 'analog_correction_utility_max', 1.0)
        )
        self.use_analog_adaptive_residual_mixer = bool(
            getattr(args, 'use_analog_adaptive_residual_mixer', 0)
        )
        self.analog_adaptive_mixer_hidden = int(
            getattr(args, 'analog_adaptive_mixer_hidden', 16)
        )
        self.analog_adaptive_mixer_node_rank = int(
            getattr(args, 'analog_adaptive_mixer_node_rank', 2)
        )
        self.analog_adaptive_mixer_max = float(
            getattr(args, 'analog_adaptive_mixer_max', 0.25)
        )
        active_residual_adapters = sum([
            self.use_node_linear_dynamics,
            self.use_regime_gated_residual,
            self.use_multires_block_residual,
            self.use_seasonal_drift_residual,
            self.use_analog_retrieval_residual,
            self.use_analog_lowrank_retrieval_residual,
            self.use_analog_node_horizon_retrieval_residual,
        ])
        if active_residual_adapters > 1:
            raise ValueError(
                'Residual adapters are mutually exclusive single-variable '
                'experiments'
            )
        analog_primary_on = (
            self.use_analog_lowrank_retrieval_residual
            or self.use_analog_node_horizon_retrieval_residual
        )
        if (
            self.use_analog_temporal_lowrank_decoder
            and not self.use_analog_lowrank_retrieval_residual
        ):
            raise ValueError(
                'analog temporal low-rank decoding is a refinement of the '
                'validated low-rank retrieval parent and requires '
                'use_analog_lowrank_retrieval_residual=1'
            )
        if self.use_analog_temporal_gap_residual and not (
            self.use_analog_lowrank_retrieval_residual
            and self.use_analog_exact_match_residual
        ):
            raise ValueError(
                'analog temporal gap residual requires the validated '
                'low-rank + exact-history retrieval parent'
            )
        if (
            self.use_analog_exact_match_residual
            and not analog_primary_on
        ):
            raise ValueError(
                'analog_exact_match_residual is a stage-2 refinement and '
                'requires one primary analog retrieval adapter'
            )
        if self.use_analog_candidate_reranker and not (
            self.use_analog_lowrank_retrieval_residual
            and self.use_analog_exact_match_residual
        ):
            raise ValueError(
                'analog_candidate_reranker refines the validated final model '
                'and requires both the low-rank and exact-history adapters'
            )
        if (
            self.use_analog_direct_prediction
            and not self.use_analog_candidate_reranker
        ):
            raise ValueError(
                'analog direct prediction requires the candidate reranker'
            )
        if self.use_analog_correction_utility_gate and not (
            self.use_analog_candidate_reranker
            and self.use_analog_lowrank_retrieval_residual
        ):
            raise ValueError(
                'analog correction utility gate requires the low-rank '
                'retrieval adapter and candidate reranker'
            )
        if self.use_analog_adaptive_residual_mixer and not (
            self.use_analog_candidate_reranker
            and self.use_analog_lowrank_retrieval_residual
        ):
            raise ValueError(
                'adaptive residual mixer requires the low-rank retrieval '
                'adapter and candidate cache loader'
            )
        if self.use_analog_adaptive_residual_mixer and (
            self.use_analog_correction_utility_gate
        ):
            raise ValueError(
                'adaptive residual mixer and correction utility gate are '
                'mutually exclusive single-variable refinements'
            )
        if self.use_analog_candidate_reranker and (
            self.use_analog_weekly_retrieval_residual
            or self.use_analog_metric_retrieval_residual
        ):
            raise ValueError(
                'analog_candidate_reranker cannot be combined with another '
                'post-retrieval refinement in a single-variable experiment'
            )

        # ===== 2B: 并行节点嵌入开关（补回 MAPA 原型压缩丢失的节点身份）=====
        # 与 getSemb(Step4, 加在 MAPA 前) 不同：本嵌入加在 MAPA 之后、预测头之前，
        # 提供一条不经过 16-prototype 瓶颈的干净节点身份通道。不改 MAPA 本身。
        self.use_parallel_node_emb = bool(getattr(args, 'use_parallel_node_emb', 0))
        self.parallel_node_emb_dropout = getattr(args, 'parallel_node_emb_dropout', 0.1)
        
        # ===== RevIN (Reversible Instance Normalization) =====
        # 在原始输入上做逐样本实例归一化（沿节点/变量维 N），去除分布偏移；
        # 输出用同一组 mean/std 反归一化。与内部 norm_flag(LayerNorm) 完全不同。
        self.use_revin = bool(getattr(args, 'use_revin', 0))
        self.revin_eps = 1e-5
        self.use_signed_change_encoding = bool(getattr(
            args, 'use_signed_change_encoding', 0
        ))
        self.use_short_window_dynamics_encoding = bool(getattr(
            args, 'use_short_window_dynamics_encoding', 0
        ))
        self.use_sparse_neighbor_innovation = bool(getattr(
            args, 'use_sparse_neighbor_innovation', 0
        ))
        if self.use_sparse_neighbor_innovation:
            graph_path = getattr(args, 'neighbor_graph_path', 'none')
            if graph_path in (None, '', 'none'):
                raise ValueError('neighbor_graph_path is required')
            with np.load(graph_path, allow_pickle=False) as graph_archive:
                neighbor_indices = np.asarray(graph_archive['neighbors'], dtype=np.int64)
            if neighbor_indices.shape[0] != self.num_nodes:
                raise ValueError('neighbor graph node count does not match model')
            # Extra module construction must not change the baseline RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.sparse_neighbor_innovation = SparseNeighborInnovation(
                    neighbor_indices, self.hid_dim
                )
        self.signed_change_encoding_max_scale = float(getattr(
            args, 'signed_change_encoding_max_scale', 0.5
        ))

        # ===== 方案二：解耦常态流/残差流（base/residual decomposition）=====
        # 核心：输入减「每日该时刻的历史均值」(history_base)，让模型只学残差(突变流)；
        # 输出加回「未来12步每日该时刻的历史均值」(future_base)。均值矩阵 M[288,N]
        # 由 compute_base_mean.py 在【训练集】上、与 dataloader 相同的 global zscore 空间预计算。
        # 注：x_time[:,:,0] 即 time-of-day 整数索引(0..287)，直接查表，无需归一化还原。
        self.base_residual = bool(getattr(args, 'base_residual', 0))
        self.base_residual_gate = bool(getattr(args, 'base_residual_gate', 0))
        if self.base_residual and self.use_seasonal_drift_residual:
            raise ValueError(
                'base_residual and seasonal_drift_residual are mutually '
                'exclusive experiments'
            )
        if self.use_seasonal_drift_residual and self.use_revin:
            raise ValueError(
                'seasonal_drift_residual requires use_revin=0 because its '
                'training-split mean is defined in global-zscore space'
            )
        if self.use_analog_retrieval_residual and self.use_revin:
            raise ValueError(
                'analog_retrieval_residual requires use_revin=0 because its '
                'cache is defined in global-zscore space'
            )
        if self.use_analog_lowrank_retrieval_residual and self.use_revin:
            raise ValueError(
                'analog_lowrank_retrieval_residual requires use_revin=0 '
                'because its cache is defined in global-zscore space'
            )
        if self.use_analog_temporal_lowrank_decoder and self.use_revin:
            raise ValueError(
                'analog temporal low-rank decoding requires use_revin=0 '
                'because its cache is defined in global-zscore space'
            )
        if self.use_analog_temporal_gap_residual and self.use_revin:
            raise ValueError(
                'analog temporal gap residual requires use_revin=0 because '
                'its cache is defined in global-zscore space'
            )
        if (
            self.use_analog_node_horizon_retrieval_residual
            and self.use_revin
        ):
            raise ValueError(
                'analog_node_horizon_retrieval_residual requires '
                'use_revin=0 because its cache is defined in global-zscore '
                'space'
            )
        if (
            self.use_analog_confidence_aware_retrieval_residual
            and self.use_revin
        ):
            raise ValueError(
                'confidence-aware retrieval requires use_revin=0 because '
                'its cache is defined in global-zscore space'
            )
        if self.use_analog_weekly_retrieval_residual and self.use_revin:
            raise ValueError(
                'weekly retrieval requires use_revin=0 because its cache is '
                'defined in global-zscore space'
            )
        if self.use_analog_metric_retrieval_residual and self.use_revin:
            raise ValueError(
                'future-consistent metric retrieval requires use_revin=0 '
                'because its cache is defined in global-zscore space'
            )
        if self.use_analog_exact_match_residual and self.use_revin:
            raise ValueError(
                'analog_exact_match_residual requires use_revin=0 because '
                'its cache is defined in global-zscore space'
            )
        if self.base_residual and self.use_analog_retrieval_residual:
            raise ValueError(
                'base_residual and analog_retrieval_residual are mutually '
                'exclusive experiments'
            )
        if self.base_residual and self.use_analog_lowrank_retrieval_residual:
            raise ValueError(
                'base_residual and analog_lowrank_retrieval_residual are '
                'mutually exclusive experiments'
            )
        if (
            self.base_residual
            and self.use_analog_node_horizon_retrieval_residual
        ):
            raise ValueError(
                'base_residual and analog_node_horizon_retrieval_residual '
                'are mutually exclusive experiments'
            )
        if (
            self.base_residual
            and self.use_analog_confidence_aware_retrieval_residual
        ):
            raise ValueError(
                'base_residual and confidence-aware retrieval are mutually '
                'exclusive experiments'
            )
        if self.base_residual and self.use_analog_weekly_retrieval_residual:
            raise ValueError(
                'base_residual and weekly retrieval are mutually exclusive '
                'experiments'
            )
        if self.base_residual and self.use_analog_metric_retrieval_residual:
            raise ValueError(
                'base_residual and future-consistent metric retrieval are '
                'mutually exclusive experiments'
            )
        if self.base_residual:
            _bm_path = getattr(args, 'base_mean_path', None)
            assert _bm_path is not None, "base_residual=True 必须提供 base_mean_path"
            # M: (slice_size_per_day=288, num_nodes)，注册为 buffer 随模型 .to(device)
            _bm = torch.load(_bm_path, map_location='cpu')
            if _bm.dim() == 1:
                _bm = _bm.unsqueeze(0)  # 兼容 (N,) 旧格式
            self.register_buffer('base_mean', _bm.float())
            # 可选动态门控：学「何时信统计基准、何时信模型残差」
            if self.base_residual_gate:
                self.base_gate = nn.Sequential(
                    nn.Linear(self.predict_length, self.predict_length),
                    nn.Sigmoid()
                )

        # 季节漂移残差需要同一份 base_mean（TOD×传感器均值），但在 base_residual
        # 关闭时也要可用。注册为 persistent=False，使其不进 state_dict，从而不破坏
        # 冻结父模型的 strict 检查（父 ckpt 只缺 seasonal_drift_residual 的可学习标量）。
        if self.use_seasonal_drift_residual and not self.base_residual:
            _sd_bm_path = getattr(args, 'base_mean_path', None)
            assert _sd_bm_path is not None, \
                "use_seasonal_drift_residual=True 必须提供 base_mean_path"
            _sd_bm = torch.load(_sd_bm_path, map_location='cpu')
            if _sd_bm.dim() == 1:
                _sd_bm = _sd_bm.unsqueeze(0)
            self.register_buffer('base_mean', _sd_bm.float(), persistent=False)

        # ===== Node Dropout（节点/传感器级 dropout，训练时随机屏蔽整条传感器序列）=====
        # 与现有 6 个 embedding/attention dropout 完全正交：在「原始输入 x」上按节点整条屏蔽，
        # 迫使模型不去依赖单一传感器、而从空间邻域重建该传感器流量。
        # 仅训练时生效（依赖 self.training）；eval/测试时 self.training=False，自动跳过、保留全部节点。
        self.use_node_dropout = bool(getattr(args, 'use_node_dropout', 0))
        self.node_dropout_p = getattr(args, 'node_dropout_p', 0.1)
        
        assert self.att_type in ['proxy', 'prob', 'full', 'difformer']
        assert self.norm_flag in ['none', 'first', 'after']
        
        # ===== 数据编码层 =====
        # 将输入数据 (x 和 latestX) 编码到隐藏空间
        self.data_encoding = DataEncoding(self.in_dim, self.hid_dim, True, activation=self.activation_data)
        if self.use_signed_change_encoding:
            # Do not advance the RNG used by any existing parent module.
            with torch.random.fork_rng(devices=[]):
                self.signed_change_encoding = SignedChangeEncoding(
                    in_dim=self.in_dim,
                    hid_dim=self.hid_dim,
                    activation=self.activation_enc,
                    max_scale=self.signed_change_encoding_max_scale,
                )
        if self.use_short_window_dynamics_encoding:
            with torch.random.fork_rng(devices=[]):
                self.short_window_dynamics_encoding = ShortWindowDynamicsEncoding(
                    hid_dim=self.hid_dim,
                    max_scale=float(getattr(
                        args, 'short_window_dynamics_encoding_max_scale', 0.5
                    )),
                )
        self.x_dropout = nn.Dropout(p=self.emb_dropout)
        
        # ===== 时间嵌入层 =====
        # 计算时间-of-day 和 day-of-week 嵌入
        self.getTemb = getTimeEmbedding(self.hid_dim, self.slice_size_per_day, self.te_emb_dropout)
        
        # ===== 全时滞嵌入层（FTLE）=====
        # 替代原始的 getTimeLagEmbedding（仅使用最后一个时间步作为参考）
        if self.use_time_lag_embedding:
            if self.use_ftle:
                self.ftle = FTLE(
                    self.hid_dim, emb_dropout=self.tle_emb_dropout
                )
            else:
                # use_ftle=0 keeps the original last-reference TLE unless the
                # independent use_time_lag_embedding switch is also disabled.
                self.get_xTimeLagEmb = getTimeLagEmbedding(
                    self.hid_dim, self.hid_dim, self.tle_emb_dropout
                )
        
        # ===== 空间嵌入层 =====
        # 计算每个节点的嵌入（可学习）
        self.getSemb = getSpatialEmbedding(self.hid_dim, self.num_nodes, self.se_emb_dropout)

        # ===== 2B: 并行节点嵌入（补回 MAPA 原型压缩丢失的节点身份）=====
        # 与 getSemb(Step4, 加在 MAPA 前) 不同：本嵌入加在 MAPA 之后、预测头之前，
        # 提供一条不经过 16-prototype 瓶颈的干净节点身份通道。不改 MAPA 本身。
        # 区别于所有失败实验：node_mix(Linear 变换)/AGCRN 替换/graph_conv 融合/mapa_bypass。
        if self.use_parallel_node_emb:
            # 每个节点一个独立可学习向量 (N, hid_dim)，广播到 (B, T, N, hid_dim)
            self.parallel_node_emb = nn.Parameter(torch.zeros(self.num_nodes, self.hid_dim))
            nn.init.xavier_uniform_(self.parallel_node_emb)
            self.parallel_node_emb_drop = nn.Dropout(self.parallel_node_emb_dropout)
        
        # ===== 多尺度时间卷积层（MS-TCN）或 ModernTCN =====
        # 替代原始的单一尺度时间卷积；tcn_type 控制具体骨干
        if self.useTCN:
            self.tcn_type = getattr(args, 'tcn_type', 'ms_tcn')
            if self.tcn_type == 'original_tcn':
                self.time_conv = OriginalTCN(self.hid_dim, self.tau)
            elif self.tcn_type == 'modern_tcn':
                # ModernTCN 式：大核时间卷积 + 节点维混合 + 通道 FFN，打破节点独立假设
                self.time_conv = ModernTCNBlock(
                    hid_dim=self.hid_dim,
                    num_nodes=self.num_nodes,
                    t_kernel=11,
                    activation=self.activation_enc
                )
                # 注：stack_blocks 仅对 ms_tcn 生效；modern_tcn 下忽略堆叠（按单次处理）
            else:
                if hasattr(args, 'tau_list') and args.tau_list is not None:
                    # 使用用户指定的 tau_list
                    tau_list = args.tau_list
                else:
                    # 默认使用 [3, 5, 7]（短/中/长期时间模式）
                    tau_list = [3, 5, 7]
                ms_tcn_dilations = getattr(args, 'ms_tcn_dilations', None)
                if not self.stack_blocks:
                    self.time_conv = MS_TCN(
                        self.hid_dim,
                        tau_list=tau_list,
                        activation=self.activation_enc,
                        dilation_list=ms_tcn_dilations,
                        num_nodes=self.num_nodes,
                        node_mix=getattr(args, 'ms_tcn_node_mix', False)
                    )
                else:
                    # 堆叠模式：每层一个独立 MS-TNN 实例（独立参数），块级 LayerNorm 沿 C 维
                    self.tcn_blocks = nn.ModuleList([
                        MS_TCN(
                            self.hid_dim,
                            tau_list=tau_list,
                            activation=self.activation_enc,
                            dilation_list=ms_tcn_dilations,
                            num_nodes=self.num_nodes,
                            node_mix=getattr(args, 'ms_tcn_node_mix', False)
                        ) for _ in range(self.num_layers)
                    ])
                    self.tcn_norm = nn.ModuleList([nn.LayerNorm(self.hid_dim) for _ in range(self.num_layers)])
                    self.spatial_norm = nn.ModuleList([nn.LayerNorm(self.hid_dim) for _ in range(self.num_layers)])
        
        # ===== 图卷积模块（保留原始功能）=====
        if self.use_graph_conv:
            # 获取 Pearson 邻接矩阵（如果存在）
            pearson_adj = getattr(args, 'pearson_adj', None)

            if self.graph_conv_type == 'adaptive':
                # 自适应图卷积（可选融合 Pearson 先验图）
                use_static = pearson_adj is not None
                self.graph_conv = AdaptiveGraphConv(
                    hid_dim=self.hid_dim,
                    num_nodes=self.num_nodes,
                    use_adaptive=True,
                    use_static=use_static,
                    static_adj=pearson_adj,
                    fusion_type='gate'
                )
            elif self.graph_conv_type == 'multiscale':
                # 多尺度图卷积（可选使用 Pearson 图作为基础）
                self.graph_conv = MultiScaleGraphConv(
                    hid_dim=self.hid_dim,
                    num_nodes=self.num_nodes,
                    k_hops=self.k_hops,
                    use_adaptive=True,
                    static_adj=pearson_adj
                )
            else:
                raise ValueError(f"Unknown graph_conv_type: {self.graph_conv_type}")

            self.graph_conv_dropout = nn.Dropout(p=self.enc_dropout)
        
        # ===== 空间聚合层（统一控制逻辑）=====
        # 设计原则：use_mapa 是 MAPA 模块的唯一开关，与 att_type 解耦
        #   - use_mapa=1: 始终使用 MAPA_EncoderLayer（记忆增强原型注意力），att_type 不再影响类型
        #   - use_mapa=0: 使用 Full_Prob_EncoderLayer（att_type 控制 full/prob）
        # 这样 FTLE+MS-TCN+MAPA 与 FTLE+MS-TCN+Full 是公平的模块级对照
        # spatial_type='agcrn' 时：用 AGCRNSpatial 直接替换 MAPA（优先级最高）
        if self.spatial_type == 'original_spa':
            self.spatial_agg_list = nn.ModuleList([
                EncoderLayer(
                    self.M, self.hid_dim, n_heads=self.n_heads,
                    num_nodes=self.num_nodes, norm_flag=self.norm_flag,
                    query_cross_time=self.query_cross_time,
                    dropout=self.enc_dropout,
                    dynamic_proxy=self.original_spa_dynamic_proxy,
                    att_dropout=self.att_dropout,
                    return_att=self.return_att,
                    activation=self.activation_enc,
                ) for _ in range(self.num_layers)
            ])
        elif self.spatial_type == 'agcrn':
            # AGCRN 式空间聚合：节点嵌入 -> 自适应邻接 -> 图卷积（替换 MAPA 的 prototype 压缩）
            pearson_adj = getattr(args, 'pearson_adj', None)
            self.spatial_agg_list = nn.ModuleList([
                AGCRNSpatial(
                    self.hid_dim, self.num_nodes,
                    dropout=self.enc_dropout,
                    static_adj=pearson_adj, use_static=True,
                    activation=self.activation_enc
                ) for _ in range(self.num_layers)
            ])
        elif self.use_mapa:
            # 使用 MAPA_EncoderLayer（记忆增强原型注意力）
            # bypass_alpha：给 MAPA 内部加一条残差绕过（output = MAPA(x) + alpha*x），
            # 干净验证"16→307 原型压缩"是否是真瓶颈。
            _bypass_alpha = float(getattr(args, 'mapa_bypass_alpha', 0.0))
            self.spatial_agg_list = nn.ModuleList([
                MAPA_EncoderLayer(
                    self.M, self.hid_dim, n_heads=self.n_heads, num_nodes=self.num_nodes,
                    norm_flag=self.norm_flag, query_cross_time=self.query_cross_time,
                    dropout=self.enc_dropout, att_dropout=self.att_dropout,
                    return_att=self.return_att, activation=self.activation_enc,
                    num_prototypes=self.num_prototypes,
                    bypass_alpha=_bypass_alpha
                ) for _ in range(self.num_layers)
            ])
        elif self.att_type == 'difformer':
            # 保留 DIFFormer 作为可选分支（向后兼容）
            self.spatial_agg_list = nn.ModuleList(
            [ DIFFormer_EncoderLayer(self.hid_dim, n_heads=self.n_heads, norm_flag=self.norm_flag,
                                        query_cross_time=self.query_cross_time, dropout=self.enc_dropout, return_att=self.return_att, activation=self.activation_enc) for _ in range(self.num_layers)]
            )
        else:
            # use_mapa=0 且非 difformer：使用 Full_Prob_EncoderLayer
            # att_type 仅控制 Full/Prob（默认 full）
            att_type_full = self.att_type if self.att_type in ['full', 'prob'] else 'full'
            self.spatial_agg_list = nn.ModuleList([
                Full_Prob_EncoderLayer(self.hid_dim, n_heads=self.n_heads,
                dropout=self.enc_dropout, att_type=att_type_full,
                                    norm_flag=self.norm_flag,
                                    query_cross_time=self.query_cross_time,
                                    att_dropout=self.att_dropout,
                                    return_att=self.return_att,
                                    prob_factor=self.prob_factor,
                                    activation=self.activation_enc) for _ in range(self.num_layers)])
        
        # ===== 时间维度预测层 =====
        # 将编码后的特征预测到未来时间步
        # predictor_type: 'linear' = 12个独立Linear（当前17.96版）；'richer' = 共享MLP+step_embedding（baseline 17.85）
        # 'richer_factorized' = SVD可初始化的Lite预测头，不保存冗余step embedding。
        _predictor_type = getattr(args, 'predictor_type', 'linear')
        if self.use_future_time_decoder and _predictor_type != 'richer':
            raise ValueError(
                'use_future_time_decoder=True requires predictor_type="richer"'
            )
        if _predictor_type == 'richer':
            self.time_wise_predictor = richer_predictor(
                self.num_nodes, self.input_length, self.predict_length, self.hid_dim,
                pre_dim=self.pre_dim, num_of_filters=self.d_out, activation='gelu',
                use_future_time_decoder=self.use_future_time_decoder
            )
        elif _predictor_type == 'richer_factorized':
            if self.use_future_time_decoder:
                raise ValueError(
                    'richer_factorized does not support '
                    'use_future_time_decoder=True'
                )
            self.time_wise_predictor = factorized_richer_predictor(
                self.num_nodes, self.input_length, self.predict_length,
                self.hid_dim, pre_dim=self.pre_dim,
                num_of_filters=self.d_out,
                factor_rank=int(getattr(args, 'predictor_factor_rank', 256)),
                activation='gelu',
                use_signed_increment_mixer=bool(getattr(
                    args, 'use_signed_increment_mixer', 0
                )),
                signed_increment_mixer_radius=int(getattr(
                    args, 'signed_increment_mixer_radius', 2
                )),
                signed_increment_mixer_max_scale=float(getattr(
                    args, 'signed_increment_mixer_max_scale', 0.25
                )),
            )
            if self.use_elastic_rank_router:
                self.time_wise_predictor.configure_elastic_rank(
                    int(getattr(args, 'elastic_low_rank', 64)),
                    target_high_fraction=float(getattr(
                        args, 'elastic_target_high_fraction', 0.5
                    )),
                    route_mode=str(getattr(
                        args, 'elastic_route_mode', 'batch_topk'
                    )),
                )
        elif _predictor_type == 'original_shared':
            # Exact predictor topology used by models/TLAST.py: shared MLP and
            # independent horizon projections, without a step embedding.
            self.time_wise_predictor = richer_predictor(
                self.num_nodes, self.input_length, self.predict_length,
                self.hid_dim, pre_dim=self.pre_dim,
                num_of_filters=self.d_out, activation='gelu',
                use_future_time_decoder=False, use_step_embedding=False,
            )
        elif _predictor_type == 'horizon_conditioned':
            self.time_wise_predictor = horizon_conditioned_predictor(
                self.num_nodes, self.input_length, self.predict_length,
                self.hid_dim, pre_dim=self.pre_dim,
                num_of_filters=self.d_out, activation='gelu',
            )
        elif _predictor_type == 'dlinear':
            self.time_wise_predictor = dlinear_decoder(
                self.num_nodes, self.input_length, self.predict_length,
                self.hid_dim, pre_dim=self.pre_dim
            )
        elif _predictor_type == 'cross_time_mlp':
            self.time_wise_predictor = cross_time_mlp(
                self.num_nodes, self.input_length, self.predict_length, self.hid_dim,
                pre_dim=self.pre_dim, num_of_filters=self.d_out, activation='gelu'
            )
        elif _predictor_type == 'ar':
            self.time_wise_predictor = ar_predictor(
                self.num_nodes, self.input_length, self.predict_length, self.hid_dim,
                pre_dim=self.pre_dim, num_of_filters=self.d_out, activation='gelu'
            )
        else:
            self.time_wise_predictor = time_wise_predictor(
                self.num_nodes, self.input_length, self.predict_length, self.hid_dim, pre_dim=self.pre_dim
            )

        if self.use_node_linear_dynamics:
            # Construct after every shared parent-model module.  This module
            # consumes no RNG and run.py preserves its explicit zero weight, so
            # all shared initialization and the initial forecast stay exactly
            # identical to the single-TLE parent.
            self.node_linear_dynamics = NodeLinearDynamicsBypass(
                num_nodes=self.num_nodes,
                input_length=self.input_length,
                predict_length=self.predict_length,
                pre_dim=self.pre_dim,
                scale=self.node_linear_dynamics_scale,
            )

        if self.use_regime_gated_residual:
            # Construct after every parent module.  The explicit zero output
            # projection makes epoch 0 exactly reproduce the loaded parent.
            self.regime_gated_residual = RegimeGatedResidualAdapter(
                input_length=self.input_length,
                predict_length=self.predict_length,
                pre_dim=self.pre_dim,
                hidden_dim=self.regime_residual_hidden,
                horizon_dim=self.regime_residual_horizon_dim,
                scale=self.regime_residual_scale,
            )

        if self.use_multires_block_residual:
            self.multires_block_residual = (
                MultiResolutionBlockResidualAdapter(
                    input_length=self.input_length,
                    predict_length=self.predict_length,
                    pre_dim=self.pre_dim,
                    block_size=self.multires_block_size,
                    hidden_dim=self.multires_block_hidden,
                    scale=self.multires_block_scale,
                )
            )

        if self.use_seasonal_drift_residual:
            # ``base_mean`` is a non-persistent model buffer loaded only from
            # the training split.  Keeping this adapter after all parent
            # modules means its zero scalar parameters cannot alter the parent
            # constructor RNG path or the epoch-0 parent forecast.
            self.seasonal_drift_residual = SeasonalDriftResidualAdapter(
                predict_length=self.predict_length,
                max_scale=self.seasonal_drift_max,
            )

        if self.use_analog_retrieval_residual:
            self.analog_retrieval_residual = (
                AnalogRetrievalResidualAdapter(
                    predict_length=self.predict_length,
                    max_blend=self.analog_retrieval_max,
                )
            )

        if self.use_analog_lowrank_retrieval_residual:
            self.analog_lowrank_retrieval_residual = (
                AnalogLowRankRetrievalResidualAdapter(
                    num_nodes=self.num_nodes,
                    predict_length=self.predict_length,
                    rank=self.analog_lowrank_rank,
                    max_blend=self.analog_retrieval_max,
                )
            )

        if self.use_analog_temporal_lowrank_decoder:
            self.analog_temporal_lowrank_decoder = (
                AnalogTemporalLowRankDecoder(
                    predict_length=self.predict_length,
                    rank=self.analog_temporal_decoder_rank,
                    max_scale=self.analog_temporal_decoder_max_scale,
                )
            )

        if self.use_analog_temporal_gap_residual:
            self.analog_temporal_gap_residual = (
                AnalogTemporalGapResidualAdapter(
                    num_nodes=self.num_nodes,
                    predict_length=self.predict_length,
                    rank=self.analog_temporal_gap_rank,
                    max_scale=self.analog_temporal_gap_max,
                )
            )

        if self.use_analog_node_horizon_retrieval_residual:
            self.analog_node_horizon_retrieval_residual = (
                AnalogNodeHorizonRetrievalResidualAdapter(
                    num_nodes=self.num_nodes,
                    predict_length=self.predict_length,
                    max_blend=self.analog_retrieval_max,
                )
            )

        if self.use_analog_confidence_aware_retrieval_residual:
            self.analog_confidence_aware_retrieval_residual = (
                AnalogConfidenceAwareRetrievalResidualAdapter(
                    predict_length=self.predict_length,
                    max_adjustment=self.analog_confidence_max,
                    num_nodes=self.num_nodes,
                    node_rank=getattr(self, 'analog_confidence_node_rank', 0),
                    feature_dim=getattr(
                        self, 'analog_confidence_feature_dim', 3
                    ),
                )
            )

        if self.use_analog_weekly_retrieval_residual:
            self.analog_weekly_retrieval_residual = (
                AnalogLowRankRetrievalResidualAdapter(
                    num_nodes=self.num_nodes,
                    predict_length=self.predict_length,
                    rank=self.analog_weekly_rank,
                    max_blend=self.analog_weekly_retrieval_max,
                )
            )

        if self.use_analog_metric_retrieval_residual:
            self.analog_metric_retrieval_residual = (
                AnalogLowRankRetrievalResidualAdapter(
                    num_nodes=self.num_nodes,
                    predict_length=self.predict_length,
                    rank=self.analog_metric_rank,
                    max_blend=self.analog_metric_retrieval_max,
                )
            )

        if self.use_analog_exact_match_residual:
            self.analog_exact_match_residual = (
                AnalogExactMatchResidualAdapter(
                    predict_length=self.predict_length,
                    max_blend=self.analog_exact_match_max,
                )
            )

        if self.use_analog_query_conditioned_residual:
            if self.use_analog_code_domain_aggregation:
                raise ValueError(
                    'Query-conditioned residual currently requires full-future '
                    'candidate payloads; disable code-domain aggregation.'
                )
            self.analog_query_conditioned_residual = (
                AnalogQueryConditionedResidual(
                    num_nodes=self.num_nodes,
                    predict_length=self.predict_length,
                    feature_dim=self.analog_candidate_feature_dim,
                    hidden_dim=self.analog_query_conditioned_hidden,
                    max_adjustment=self.analog_query_conditioned_max,
                )
            )

        if self.use_analog_candidate_reranker:
            # Constructed after the complete validated parent (including the
            # rank-2 and exact-history adapters).  Frozen-parent loading must
            # therefore miss only this module's parameters.
            self.analog_candidate_reranker = (
                AnalogCandidateHorizonReranker(
                    num_nodes=self.num_nodes,
                    predict_length=self.predict_length,
                    feature_dim=self.analog_candidate_feature_dim,
                    disagreement_dim=self.analog_candidate_disagreement_dim,
                    hidden_dim=self.analog_candidate_hidden,
                    code_domain=self.use_analog_code_domain_aggregation,
                    code_rank=self.analog_candidate_code_rank,
                    temporal_mean=self.analog_candidate_code_mean,
                    temporal_basis=self.analog_candidate_code_basis,
                    decode_candidates=self.analog_code_domain_decode_candidates,
                    level_align=bool(getattr(args, 'analog_candidate_level_align', 0)),
                    candidate_parent_residual_features=(
                        self.analog_candidate_parent_residual_features
                    ),
                )
            )

        if self.use_analog_correction_utility_gate:
            self.analog_correction_utility_gate = AnalogCorrectionUtilityGate(
                num_nodes=self.num_nodes,
                predict_length=self.predict_length,
                hidden_dim=self.analog_correction_utility_hidden,
                node_rank=self.analog_correction_utility_node_rank,
                max_suppression=self.analog_correction_utility_max,
            )

        if self.use_analog_adaptive_residual_mixer:
            self.analog_adaptive_residual_mixer = AnalogAdaptiveResidualMixer(
                num_nodes=self.num_nodes,
                predict_length=self.predict_length,
                candidate_feature_dim=self.analog_candidate_feature_dim,
                hidden_dim=self.analog_adaptive_mixer_hidden,
                node_rank=self.analog_adaptive_mixer_node_rank,
                max_scale=self.analog_adaptive_mixer_max,
            )

        if (
            self.use_time_lag_embedding
            and self.use_ftle
            and self.ftle_mode == 'gated_blend'
        ):
            # Add this branch after all baseline modules and restore the constructor
            # RNG state. The historical global Xavier pass therefore leaves every
            # shared baseline parameter bit-identical, while initializing this
            # additional branch last.
            rng_state = torch.random.get_rng_state()
            self.gated_last_lag = getTimeLagEmbedding(
                self.hid_dim, self.hid_dim, self.tle_emb_dropout
            )
            torch.random.set_rng_state(rng_state)
            self.ftle_blend = nn.Parameter(torch.zeros(()))

        if (
            self.use_time_lag_embedding
            and not self.use_ftle
            and self.single_tle_mode == 'selective'
        ):
            # Construct the only new parameter after every shared baseline
            # module. PyTorch enumerates direct root parameters before child
            # modules, but run.py's global Xavier pass skips this 1-D zero
            # tensor without consuming RNG. All shared parameters therefore
            # keep the same initialization as last-reference TLE, and the
            # initial prediction is exactly identical.
            self.selective_lag_gates = nn.Parameter(
                torch.zeros(self.input_length - 1)
            )

        if self.use_volatility_delta_scale:
            # Construct after every shared baseline module. Restoring the RNG
            # state plus preserve_initialization keeps shared parameters
            # bit-identical and protects the explicit zero projection from
            # run.py's global Xavier pass.
            rng_state = torch.random.get_rng_state()
            self.volatility_delta_scaler = VolatilityConditionedDeltaScale(
                self.predict_length,
                hidden_dim=self.volatility_delta_hidden,
                max_adjustment=self.volatility_delta_max,
            )
            torch.random.set_rng_state(rng_state)

        if self.use_student_risk_head:
            self.student_risk_head = StudentRiskHead(
                self.slice_size_per_day,
                hidden_dim=self.student_risk_hidden,
            )
        
        # ===== 正交损失权重 =====
        # 用于训练时平衡主损失和正交损失
        self.lambda_ortho = self.lambda_ortho

    def _get_time_lag_embedding(self, x_time_emb):
        """Return the configured time-lag embedding.

        Selective single-reference TLE starts exactly from the last-reference
        branch. Each zero-initialized scalar gate may add only the correction
        contributed by one earlier reference. The correction calls reuse the
        same TLE MLP, adding no alternative lag encoder.
        """
        if not self.use_time_lag_embedding:
            return torch.zeros_like(x_time_emb)

        if self.use_ftle:
            x_time_lag_emb = self.ftle(x_time_emb)
            if self.ftle_mode == 'gated_blend':
                last_lag = self.gated_last_lag(
                    x_time_emb[:, -1:, :, :], x_time_emb
                )
                blend = torch.tanh(self.ftle_blend)
                x_time_lag_emb = x_time_lag_emb + blend * (
                    last_lag - x_time_lag_emb
                )
            return x_time_lag_emb

        last_lag = self.get_xTimeLagEmb(
            x_time_emb[:, -1:, :, :], x_time_emb
        )
        if self.single_tle_mode == 'last':
            return last_lag

        num_earlier = x_time_emb.shape[1] - 1
        if num_earlier != self.selective_lag_gates.numel():
            raise ValueError(
                'Selective TLE expected '
                f'{self.selective_lag_gates.numel() + 1} input steps, got '
                f'{x_time_emb.shape[1]}'
            )
        if num_earlier == 0:
            return last_lag

        gates = torch.tanh(self.selective_lag_gates)
        correction = torch.zeros_like(last_lag)

        # Earlier-reference dropout must not advance the RNG used by shared
        # downstream modules. fork_rng restores CPU/CUDA states on exit, while
        # retaining gradients through the reference corrections and gates.
        devices = (
            [x_time_emb.device.index]
            if x_time_emb.is_cuda and x_time_emb.device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=devices):
            for ref_idx in range(num_earlier):
                reference_lag = self.get_xTimeLagEmb(
                    x_time_emb[:, ref_idx:ref_idx + 1, :, :], x_time_emb
                )
                correction = correction + gates[ref_idx] * (
                    reference_lag - last_lag
                )
        return last_lag + correction / num_earlier
    
    def forward(self, x):
        """
        前向传播：将输入数据通过所有模块，生成预测结果
        
        参数：
            x: tuple (x, x_time), (x, x_time, y_time),
               (x, x_time, y_time, analog_retrieval), the latter plus
               (exact_retrieval, exact_available), or the complete form plus
               (candidate_future, candidate_available, candidate_features)
                - x: (B, T, N, in_dim)  # 输入交通数据
                - x_time: (B, T, 2)  # 时间特征（time-of-day 和 day-of-week），列0=时刻索引(0..287)
                - y_time: (B, T', 2)  # future_time_decoder/base_residual 使用的已知未来日历特征
        
        返回：
            main_output: (B, T', N, pre_dim)  # 预测结果
            total_ortho_loss: scalar tensor  # 总正交损失（如果使用了 MAPA）
                                                    # 如果没有使用 MAPA，则返回 0
        """
        # 解包输入（兼容旧调用；第四项仅供 train-memory analog 分支使用）
        analog_retrieval = None
        exact_retrieval = None
        exact_available = None
        candidate_future = None
        candidate_available = None
        candidate_features = None
        if isinstance(x, (list, tuple)):
            if len(x) == 9:
                (
                    x, x_time, y_time, analog_retrieval,
                    exact_retrieval, exact_available,
                    candidate_future, candidate_available,
                    candidate_features,
                ) = x
            elif len(x) == 6:
                (
                    x, x_time, y_time, analog_retrieval,
                    exact_retrieval, exact_available,
                ) = x
            elif len(x) == 4:
                x, x_time, y_time, analog_retrieval = x
            elif len(x) == 3:
                x, x_time, y_time = x
            elif len(x) == 2:
                x, x_time = x
                y_time = None
            else:
                raise ValueError(
                    f'Expected 2, 3, 4, 6, or 9 model inputs, got {len(x)}'
                )
        else:
            x_time = None
            y_time = None
        B, T, N, _ = x.shape
        assert x.shape[-1] == self.in_dim
        risk_history = x[..., :self.pre_dim]
        if self.use_volatility_delta_scale:
            volatility_history = x
            volatility_last = x[:, -1:, :, :self.pre_dim]

        # ===== 方案二：输入减「历史每日均值」(history_base) =====
        # 让模型从「残差(突变流)」学习，剥离强周期性。base_mean 在训练集 global zscore 空间预计算。
        if self.base_residual and x_time is not None:
            tod_in = x_time[:, :, 0].long().clamp_(0, self.base_mean.shape[0] - 1)  # (B, T)
            history_base = self.base_mean[tod_in]                                  # (B, T, N)
            x = x - history_base.unsqueeze(-1)                                     # (B, T, N, 1)

        # ===== Node Dropout（节点级整条屏蔽，仅训练时生效）=====
        # 对每个 batch 样本，按概率 node_dropout_p 随机屏蔽部分传感器（整条时间序列置 0），
        # 迫使模型从空间邻域重建被屏蔽传感器的流量，降低对单一传感器的过拟合。
        # 注意：只屏蔽流量 x，不碰时间特征 x_time（其按结构共享，不应被屏蔽）；
        # eval 时 self.training=False，跳过 → 与基线完全一致的推理路径。
        # 被全 0 屏蔽的节点在后续 RevIN 中 std→eps，归一化后仍为 0，不产生 NaN。
        if self.use_node_dropout and self.training:
            # node_mask: (B, 1, N, 1)，1=保留，0=屏蔽
            node_mask = (torch.rand(B, 1, N, 1, device=x.device) > self.node_dropout_p).float()
            x = x * node_mask

        # ===== RevIN (Reversible Instance Normalization) =====
        # 在原始输入 x 上沿节点/变量维(N)做逐样本实例归一化，存储 mean/std 供输出反归一化。
        # 数据进入模型前已全局 z-score，RevIN 叠加其上、进一步消除逐样本/逐时刻分布偏移。
        # 注：仅归一化流量 x，不碰时间特征 x_time；要求 input_length==predict_length。
        if self.use_revin:
            # RevIN statistics are per sample/node/feature over the historical
            # time axis. The previous implementation normalized across nodes,
            # which is not reversible instance normalization for node series.
            self._revin_mean = x.mean(dim=1, keepdim=True)  # (B, 1, N, C)
            self._revin_std = (
                x.std(dim=1, keepdim=True, unbiased=False) + self.revin_eps
            )
            x = (x - self._revin_mean) / self._revin_std

        # ===== Step 1: 数据编码 =====
        # 将当前时间步的特征与最新时间步的特征拼接，然后编码到隐藏空间
        latestX = x[:, -1:, :, :].repeat([1, self.input_length, 1, 1])  # (B, T, N, in_dim)
        data = self.data_encoding(x, latestX)  # (B, T, N, hid_dim)
        if self.use_signed_change_encoding:
            data = data + self.signed_change_encoding(x)
        if self.use_short_window_dynamics_encoding:
            data = data + self.short_window_dynamics_encoding(x)
        data = self.x_dropout(data)
        
        # ===== Step 2: 时间嵌入 =====
        # 添加时间-of-day 和 day-of-week 嵌入
        x_time_emb = self.getTemb(x_time)  # (B, T, N, hid_dim)
        data = data + x_time_emb  # 残差连接
        
        # ===== Step 3: 全时滞嵌入（FTLE）=====
        # 替代原始的单参考时滞嵌入
        x_time_lag_emb = self._get_time_lag_embedding(x_time_emb)
        data = data + x_time_lag_emb  # 残差连接
        
        # ===== Step 4: 空间嵌入 =====
        # 添加节点（传感器）嵌入
        x_spa_emb = self.getSemb(data)  # (1, 1, N, hid_dim)
        data = data + x_spa_emb  # 残差连接（自动广播到 B 和 T）
        if self.use_sparse_neighbor_innovation:
            data = data + self.sparse_neighbor_innovation(x)
        
        # ===== Step 5: 多尺度时间卷积（MS-TCN）=====
        # 捕获多尺度时间模式
        # 堆叠模式（实验A）下，MS-TCN 被移入每个时空块内部，此处跳过单次调用
        _stack_mode = self.stack_blocks and hasattr(self, 'tcn_blocks')
        if self.useTCN and not _stack_mode:
            # 转置为 (B, C, N, T) 以应用 2D 卷积
            data = data.transpose(1, 3)  # (B, T, N, C) -> (B, C, N, T)
            data = self.time_conv(data)  # (B, C, N, T)
            # 转置回 (B, T, N, C)
            data = data.transpose(1, 3)  # (B, C, N, T) -> (B, T, N, C)
            assert data.shape[1] == self.input_length  # 确保时间维度 T 不变
        
        # ===== Step 6: 图卷积模块（保留原始功能）=====
        if self.use_graph_conv:
            graph_out = self.graph_conv(data)
            data = data + self.graph_conv_dropout(graph_out)  # 残差连接
        
        # ===== Step 7: 空间聚合 或 堆叠时空块（实验A）=====
        total_ortho_loss = 0  # 累计正交损失
        A_list = []  # 存储注意力权重（如果 return_att=True）
        
        if _stack_mode:
            # 堆叠时空块：[MS-TCN -> spatial] × num_layers，每块带残差 + LayerNorm（沿 C 维）
            for i in range(self.num_layers):
                # --- 时间分支：MS-TCN + 残差 + LayerNorm ---
                x_bt = data.transpose(1, 3)              # (B, T, N, C) -> (B, C, N, T)
                t_out = self.tcn_blocks[i](x_bt)          # (B, C, N, T)
                x_bt = x_bt + t_out                       # 残差连接
                x_bt = x_bt.transpose(1, 3)              # (B, C, N, T) -> (B, T, N, C)
                data = self.tcn_norm[i](x_bt)             # LayerNorm 沿 C 维
                # --- 空间分支：spatial_agg + 残差 + LayerNorm ---
                s = self.spatial_agg_list[i](data)
                if self.spatial_type == 'agcrn':
                    data = data + s                       # AGCRNSpatial 直接返回 (B, T, N, C)
                elif self.use_mapa:
                    if self.return_att:
                        s, ortho_loss, A = s
                        A_list.append(A)
                    else:
                        s, ortho_loss = s
                    total_ortho_loss += ortho_loss
                    data = data + s
                else:
                    s, A = s
                    A_list.append(A)
                    data = data + s
                data = self.spatial_norm[i](data)         # LayerNorm 沿 C 维
            assert data.shape[1] == self.input_length     # 确保时间维度 T 不变
        else:
            # 原逻辑：单次 MS-TCN 已在 Step5 完成，此处循环 spatial_agg + 外层 skip 连接
            skip = data  # 跳过连接
            for i in range(self.num_layers):
                if self.spatial_type == 'agcrn':
                    # AGCRNSpatial 直接返回 (B, T, N, C)，无正交损失
                    data = self.spatial_agg_list[i](data)
                elif self.use_mapa:
                    # MAPA_EncoderLayer 返回格式：
                    #   - 如果 return_att=True: (output, ortho_loss, A_list)
                    #   - 如果 return_att=False: (output, ortho_loss)
                    if self.return_att:
                        data, ortho_loss, A = self.spatial_agg_list[i](data)
                        A_list.append(A)
                    else:
                        data, ortho_loss = self.spatial_agg_list[i](data)
                    total_ortho_loss += ortho_loss  # 累计正交损失
                else:
                    # Full_Prob_EncoderLayer / DIFFormer 返回 (output, A)
                    data, A = self.spatial_agg_list[i](data)
                    A_list.append(A)
            data += skip  # 跳过连接
        
        # ===== 2B: 并行节点嵌入（补回 MAPA 原型压缩丢失的节点身份）=====
        # MAPA 把 307 节点压成 num_prototypes(16) 个 prototype 再还原，节点级身份
        # 经"共享原型基"重建后被平均掉；getSemb(Step4) 虽在 MAPA 前加过节点嵌入，
        # 但同样被原型压缩模糊。故在 MAPA 之后、预测头之前，并行加一份【独立】可学习
        # 节点嵌入(broadcast 到 B,T)，给预测头一条不经过原型瓶颈的干净节点身份通道。
        # 不改 MAPA 本身（与所有失败实验的 node_mix/AGCRN 替换/graph_conv 融合均不同）。
        if self.use_parallel_node_emb:
            pne = self.parallel_node_emb.unsqueeze(0).unsqueeze(1)   # (N,d) -> (1,1,N,d)
            pne = self.parallel_node_emb_drop(pne)                   # 训练时 dropout
            data = data + pne
        
        # ===== Step 8: 时间维度预测 =====
        # 将编码后的特征预测到未来时间步
        if self.use_future_time_decoder:
            if y_time is None:
                raise ValueError(
                    'y_time is required when use_future_time_decoder=True'
                )
            future_time_emb = self.getTemb(y_time)  # (B, T', 1, hid_dim)
            main_output = self.time_wise_predictor(data, future_time_emb)
        else:
            if self.use_elastic_rank_router:
                elastic_probability = self.elastic_rank_router(
                    x[..., :self.pre_dim],
                    temperature=self.elastic_rank_temperature,
                )
                main_output = self.time_wise_predictor(
                    data, elastic_probability=elastic_probability
                )
            else:
                main_output = self.time_wise_predictor(data)

        # A conservative local-dynamics correction learned jointly on the
        # training set.  It is part of the raw model forward, not validation
        # calibration, and uses no future traffic values at inference.
        if self.use_node_linear_dynamics:
            main_output = main_output + self.node_linear_dynamics(
                x[..., :self.pre_dim]
            )

        if self.use_regime_gated_residual:
            parent_delta = main_output
            main_output = parent_delta + self.regime_gated_residual(
                x[..., :self.pre_dim], parent_delta
            )

        if self.use_multires_block_residual:
            parent_delta = main_output
            main_output = parent_delta + self.multires_block_residual(
                x[..., :self.pre_dim], parent_delta
            )
        
        # ===== Step 9: 残差连接 =====
        # 将最新时间步的特征作为预测的基线（类似残差连接）
        if self.input_length == self.predict_length:
            main_output += latestX
        else:
            main_output += latestX[:, 0:self.predict_length, :, :]
        
        # ===== RevIN 反归一化 =====
        # 用输入统计量把预测从归一化空间还原到 z-score 空间，再交外层 scaler 还原原量纲。
        if self.use_revin:
            main_output = main_output * self._revin_std + self._revin_mean

        # ===== 方案二：输出加回「未来每日均值」(future_base) =====
        # 模型输出的是残差预测，加回该时刻历史均值即还原为最终预测。
        # 与 history_base 同一 M 矩阵、同一 zscore 空间，保证加减口径一致。
        if self.base_residual and y_time is not None:
            tod_out = y_time[:, :, 0].long().clamp_(0, self.base_mean.shape[0] - 1)  # (B, T')
            future_base = self.base_mean[tod_out]                                   # (B, T', N)
            future_base = future_base.unsqueeze(-1)                                 # (B, T', N, 1)
            if self.base_residual_gate:
                # 动态门控：g∈(0,1) 决定「模型残差」相对「统计基准」的信任权重
                g = self.base_gate(main_output.squeeze(-1).transpose(1, 2))        # (B, N, T')
                g = g.transpose(1, 2).unsqueeze(-1)                                 # (B, T', N, 1)
                main_output = future_base + g * main_output
            else:
                main_output = main_output + future_base

        # The seasonal-drift branch operates on the parent's final
        # global-zscore forecast.  It uses only input/future calendar indices
        # and a mean calculated from idx_train; no validation/test target is
        # consulted at inference or during fitting.
        if self.use_seasonal_drift_residual:
            main_output = self.seasonal_drift_residual(
                main_output, self.base_mean, x_time, y_time
            )

        # The analog forecast is a deterministic model input constructed from
        # the training split only.  The learned horizon blend is fitted by the
        # ordinary training loss and is part of the saved raw model.
        if self.use_analog_retrieval_residual:
            if analog_retrieval is None:
                raise ValueError(
                    'analog_retrieval input is required when '
                    'use_analog_retrieval_residual=True'
                )
            main_output = self.analog_retrieval_residual(
                main_output,
                analog_retrieval[..., :main_output.shape[-1]],
            )

        primary_analog_retrieval = None
        if analog_retrieval is not None:
            primary_analog_retrieval = analog_retrieval[
                ..., :main_output.shape[-1]
            ]

        if self.use_analog_temporal_lowrank_decoder:
            if primary_analog_retrieval is None:
                raise ValueError(
                    'analog_retrieval is required by the temporal low-rank '
                    'decoder'
                )
            primary_analog_retrieval = (
                self.analog_temporal_lowrank_decoder(
                    primary_analog_retrieval
                )
            )

        if self.use_analog_candidate_reranker:
            if primary_analog_retrieval is None:
                raise ValueError(
                    'analog_retrieval is required by the candidate reranker'
                )
            if (
                candidate_future is None
                or candidate_available is None
                or candidate_features is None
                or exact_retrieval is None
                or exact_available is None
            ):
                raise ValueError(
                    'candidate tensors and exact-history tensors are required '
                    'by the candidate reranker'
                )
            if self.use_analog_query_conditioned_residual:
                # Change candidate residual content using only observable
                # query/history difference summaries.  The conditioner is
                # zero-initialized, so this path is an exact parent anchor at
                # epoch 0 and does not use the validation/test target.
                conditioned_candidate = self.analog_query_conditioned_residual(
                    candidate_future,
                    candidate_available,
                    candidate_features,
                )
                available = candidate_available.to(dtype=conditioned_candidate.dtype)
                available_count = available.sum(dim=1)
                candidate_delta = conditioned_candidate - candidate_future
                conditioned_mean_delta = (
                    candidate_delta * available
                ).sum(dim=1) / available_count.clamp_min(1.0)
                # Preserve the cached parent mean exactly when the
                # zero-initialized conditioner is inactive.  Recomputing a
                # candidate mean here would introduce a cache/fallback
                # mismatch and invalidate the frozen-parent anchor.
                primary_analog_retrieval = primary_analog_retrieval + torch.where(
                    available_count > 0,
                    conditioned_mean_delta,
                    torch.zeros_like(primary_analog_retrieval),
                )
                candidate_future = conditioned_candidate
            # Obtain the unchanged final-parent forecast used only as a causal
            # ranking feature.  The actual forecast is recomputed below after
            # the mean retrieval has been reweighted.  These two tiny adapter
            # calls do not rerun the backbone.
            frozen_parent_prediction = (
                self.analog_lowrank_retrieval_residual(
                    main_output, primary_analog_retrieval
                )
            )
            frozen_parent_prediction = self.analog_exact_match_residual(
                frozen_parent_prediction, exact_retrieval, exact_available
            )
            primary_analog_retrieval = self.analog_candidate_reranker(
                frozen_parent_prediction.detach(),
                primary_analog_retrieval,
                candidate_future,
                candidate_available,
                candidate_features,
            )

        if self.use_analog_direct_prediction:
            # Direct-retrieval control: the same reranked K-candidate analog
            # is the final prediction.  The frozen low-rank + exact path above
            # is used only to construct target-free disagreement features.
            main_output = primary_analog_retrieval

        if (
            self.use_analog_lowrank_retrieval_residual
            and not self.use_analog_direct_prediction
        ):
            if analog_retrieval is None:
                raise ValueError(
                    'analog_retrieval input is required when '
                    'use_analog_lowrank_retrieval_residual=True'
                )
            near_base_prediction = main_output
            main_output = self.analog_lowrank_retrieval_residual(
                main_output,
                primary_analog_retrieval,
            )
            if self.use_analog_adaptive_residual_mixer:
                if (
                    candidate_future is None
                    or candidate_available is None
                    or candidate_features is None
                ):
                    raise ValueError(
                        'candidate tensors are required by the adaptive '
                        'residual mixer'
                    )
                main_output = self.analog_adaptive_residual_mixer(
                    main_output,
                    candidate_future,
                    candidate_available,
                    candidate_features,
                )
            if self.use_analog_correction_utility_gate:
                main_output = self.analog_correction_utility_gate(
                    near_base_prediction,
                    main_output,
                    primary_analog_retrieval,
                    analog_retrieval,
                    self.analog_candidate_reranker.last_weights,
                )

        if self.use_analog_node_horizon_retrieval_residual:
            if analog_retrieval is None:
                raise ValueError(
                    'analog_retrieval input is required when '
                    'use_analog_node_horizon_retrieval_residual=True'
                )
            main_output = self.analog_node_horizon_retrieval_residual(
                main_output,
                primary_analog_retrieval,
            )

        # Stage-2 exact-history refinement.  It is applied after the frozen
        # primary analog parent so every unavailable element remains
        # bit-identical to that parent.  The cache contains only train-memory
        # continuations.
        if (
            self.use_analog_exact_match_residual
            and not self.use_analog_direct_prediction
        ):
            if exact_retrieval is None or exact_available is None:
                raise ValueError(
                    'exact_retrieval and exact_available are required when '
                    'use_analog_exact_match_residual=True'
                )
            main_output = self.analog_exact_match_residual(
                main_output, exact_retrieval, exact_available
            )

        # Stage-3 dual-calendar memory: the daily TOD retrieval remains the
        # frozen parent, while this zero-initialized low-rank gate may move
        # selected node/horizon predictions toward same-TOD+DOW continuations.
        if self.use_analog_weekly_retrieval_residual:
            if analog_retrieval is None:
                raise ValueError(
                    'analog_retrieval input is required when '
                    'use_analog_weekly_retrieval_residual=True'
                )
            channel_count = main_output.shape[-1]
            if analog_retrieval.shape[-1] < 2 * channel_count:
                raise ValueError(
                    'Weekly retrieval requires daily and weekly forecast '
                    'channels in analog_retrieval'
                )
            weekly_retrieval = analog_retrieval[
                ..., channel_count:2 * channel_count
            ]
            main_output = self.analog_weekly_retrieval_residual(
                main_output, weekly_retrieval
            )

        # Stage-3 future-consistent memory.  The auxiliary retrieval channel
        # is produced by a train-only supervised history encoder and contains
        # only training-split continuations.
        if self.use_analog_metric_retrieval_residual:
            if analog_retrieval is None:
                raise ValueError(
                    'analog_retrieval input is required when '
                    'use_analog_metric_retrieval_residual=True'
                )
            channel_count = main_output.shape[-1]
            if analog_retrieval.shape[-1] < 2 * channel_count:
                raise ValueError(
                    'Metric retrieval requires daily and future-consistent '
                    'forecast channels in analog_retrieval'
                )
            metric_retrieval = analog_retrieval[
                ..., channel_count:2 * channel_count
            ]
            main_output = self.analog_metric_retrieval_residual(
                main_output, metric_retrieval
            )

        # Stage-3 local reliability refinement.  This is deliberately applied
        # after the validated rank-2 + exact parent and is identity at epoch 0.
        if self.use_analog_confidence_aware_retrieval_residual:
            if analog_retrieval is None:
                raise ValueError(
                    'analog_retrieval input is required when '
                    'use_analog_confidence_aware_retrieval_residual=True'
                )
            confidence_retrieval = analog_retrieval
            if self.use_analog_candidate_reranker:
                # Gate the reranked K-candidate forecast while retaining the
                # history-distance and future-dispersion quality channels
                # computed leakage-free by the retrieval cache.
                channel_count = main_output.shape[-1]
                confidence_retrieval = torch.cat([
                    primary_analog_retrieval,
                    analog_retrieval[
                        ..., channel_count:channel_count + 2
                    ],
                ], dim=-1)
            main_output = self.analog_confidence_aware_retrieval_residual(
                main_output, confidence_retrieval, exact_available
            )

        # Final low-rank temporal calibration. Exact-history memory itself is
        # untouched; it is selected only as a train-memory reference wherever
        # available, with the reranked near-match continuation as fallback.
        if self.use_analog_temporal_gap_residual:
            if (
                primary_analog_retrieval is None
                or exact_retrieval is None
                or exact_available is None
            ):
                raise ValueError(
                    'near-match and exact-history tensors are required by '
                    'the analog temporal gap residual'
                )
            temporal_reference = torch.where(
                exact_available.bool(),
                exact_retrieval,
                primary_analog_retrieval,
            )
            main_output = self.analog_temporal_gap_residual(
                main_output, temporal_reference
            )

        # Apply in the original global-zscore space after reversible/base
        # transformations. This training-time component never reads targets or
        # future traffic values.
        if self.use_volatility_delta_scale:
            main_output = self.volatility_delta_scaler(
                main_output, volatility_last, volatility_history
            )

        if self.use_student_risk_head:
            self.last_risk_score = self.student_risk_head(
                risk_history, main_output, x_time
            )
        else:
            self.last_risk_score = None

        # ===== Step 10: 返回结果 =====
        # 将正交损失存储为模型属性（以便 run.py 访问）
        self.ortho_loss = total_ortho_loss
        
        if self.return_att:
            # 如果返回注意力权重，返回 (main_output, A_list)
            return main_output, A_list
        else:
            # 如果不返回注意力权重，返回 (main_output, None)
            return main_output, None

    def encode(self, x, x_time):
        """
        编码器前向（Step 1~6）：返回 spatial_agg_list 之后的 hidden state。
        用于 STD-MAE 预训练：复用 TLAST 编码器，不重复实现。
        返回：(hidden: (B, T, N, hid_dim), ortho_loss: scalar)
        """
        if isinstance(x, (tuple, list)):
            if len(x) < 2:
                raise ValueError('encode expects (x, x_time) or (x, x_time, y_time)')
            x_input, x_time_in = x[0], x[1]
        else:
            x_input, x_time_in = x, x_time
        if x_time_in is None:
            raise ValueError('x_time is required by the TLAST encoder')
        B, T, N, _ = x_input.shape
        # Step 1
        latestX = x_input[:, -1, :, :].unsqueeze(1).repeat(1, T, 1, 1)
        data = self.data_encoding(x_input, latestX)
        data = self.x_dropout(data)
        # Step 2
        x_time_emb = self.getTemb(x_time_in)
        data = data + x_time_emb
        # Step 3
        x_time_lag_emb = self._get_time_lag_embedding(x_time_emb)
        data = data + x_time_lag_emb
        # Step 4
        x_spa_emb = self.getSemb(data)
        data = data + x_spa_emb
        if self.use_sparse_neighbor_innovation:
            data = data + self.sparse_neighbor_innovation(x_input)
        # Step 5
        _stack_mode = self.stack_blocks and hasattr(self, 'tcn_blocks')
        if self.useTCN and not _stack_mode:
            data = data.transpose(1, 3)  # (B, T, N, C) -> (B, C, N, T)
            data = self.time_conv(data)
            data = data.transpose(1, 3)  # (B, C, N, T) -> (B, T, N, C)
        # ===== P1 修复：与主 forward 对齐，加 use_graph_conv 残差 =====
        if self.use_graph_conv:
            graph_out = self.graph_conv(data)
            data = data + self.graph_conv_dropout(graph_out)
        # Step 6
        total_ortho_loss = 0
        _stack_mode = self.stack_blocks and hasattr(self, 'tcn_blocks')
        if _stack_mode:
            # 简化版：仅取最后一层 spatial_agg 的输出作 hidden
            for i in range(self.num_layers):
                x_bt = data.transpose(1, 3)
                t_out = self.tcn_blocks[i](x_bt)
                x_bt = x_bt + t_out
                x_bt = x_bt.transpose(1, 3)
                data = self.tcn_norm[i](x_bt)
                s = self.spatial_agg_list[i](data)
                if self.spatial_type == 'agcrn':
                    data = data + s
                elif self.use_mapa:
                    if self.return_att:
                        s, ortho_loss, _ = s
                    else:
                        s, ortho_loss = s
                    total_ortho_loss += ortho_loss
                    data = data + s
                else:
                    s, A = s
                    data = data + s
                data = self.spatial_norm[i](data)
        else:
            skip = data
            for i in range(self.num_layers):
                s = self.spatial_agg_list[i](data)
                if self.spatial_type == 'agcrn':
                    data = s
                elif self.use_mapa:
                    if self.return_att:
                        s, ortho_loss, A = s
                    else:
                        s, ortho_loss = s
                    total_ortho_loss += ortho_loss
                    data = s
                else:
                    s, A = s
                    data = s
            data = data + skip
        if self.use_parallel_node_emb:
            pne = self.parallel_node_emb.unsqueeze(0).unsqueeze(1)
            data = data + self.parallel_node_emb_drop(pne)
        return data, total_ortho_loss
