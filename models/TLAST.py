import torch
import torch.nn as nn
from math import sqrt

from functools import partial
from einops import rearrange, repeat
from layers.DIFFormer import DIFFormerConv
from layers.ProbAttention import ProbAttention
from layers.GraphConv import AdaptiveGraphConv, MultiScaleGraphConv

class MultiHeadsAttention(nn.Module):
    '''
    The Attention operation
    '''
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
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

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
    '''
    Use Difformer Attention
    '''

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
    '''
    Use FullAttention or ProbAttention
    '''

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
    '''
    Use Proxy Attention
    '''

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

    def __init__(self, num_nodes, input_length, predict_length, in_dim, pre_dim,
                 num_of_filters=128, activation='gelu'):
        super(time_wise_predictor, self).__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.in_dim = in_dim
        self.pre_dim = pre_dim
        self.predict_length = predict_length
        assert activation in ['gelu', 'relu']
        self.predict_unit = nn.Sequential(nn.Linear(input_length*in_dim, num_of_filters),
                                          nn.GELU() if activation == 'gelu' else nn.ReLU())
        self.predict_unit_list = nn.ModuleList(
                [nn.Linear(num_of_filters, pre_dim) for _ in range(predict_length)])

    def forward(self, data):
        data = rearrange(data, 'b t n c -> b n (t c)')
        data = self.predict_unit(data)
        need_concat = []
        B,N,_ = data.shape
       
        for j in range(self.predict_length):
            unit_out = self.predict_unit_list[j](data)
            unit_out = rearrange(unit_out, 'b n c -> b 1 n c')
            need_concat.append(unit_out)
        final_out = torch.cat(need_concat, dim=1)
        return final_out

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
        spatial_emb = self.spatial_embedding(spatial_indexs).unsqueeze(0).unsqueeze(1) # (N, d)->(1, 1, N, d)
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


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_length = args.input_length
        self.predict_length = args.predict_length
        self.in_dim = args.in_dim
        self.pre_dim = args.pre_dim
        self.num_nodes = args.num_nodes
        self.tau = args.tau
        self.useTCN = self.tau > 0
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

        # Graph convolution parameters
        self.use_graph_conv = bool(getattr(args, 'use_graph_conv', 0))
        self.graph_conv_type = getattr(args, 'graph_conv_type', 'adaptive')
        self.k_hops = getattr(args, 'k_hops', [1, 2])

        assert self.att_type in ['proxy', 'prob', 'full', 'difformer']
        assert self.norm_flag in ['none', 'first', 'after']

        self.data_encoding = DataEncoding(self.in_dim, self.hid_dim, True, activation=self.activation_data)
        self.x_dropout = nn.Dropout(p=self.emb_dropout)
        
        self.getTemb = getTimeEmbedding(self.hid_dim, self.slice_size_per_day, self.te_emb_dropout)
        self.get_xTimeLagEmb = getTimeLagEmbedding(self.hid_dim, self.hid_dim, self.tle_emb_dropout)
        self.getSemb = getSpatialEmbedding(self.hid_dim, self.num_nodes, self.se_emb_dropout)
        
        if self.useTCN:
            assert self.tau in [2,3]
            tcn_pad_l, tcn_pad_r = self.tau-2, 1 
            self.padding = nn.ReplicationPad2d((tcn_pad_l, tcn_pad_r, 0, 0))  # x must be like (B,C,N,T)
            self.time_conv = nn.Conv2d(self.hid_dim, self.hid_dim, (1, self.tau))

        # Graph Convolution Module
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

        if self.att_type in ['full', 'prob']:
            self.spatial_agg_list = nn.ModuleList([
                Full_Prob_EncoderLayer(self.hid_dim, n_heads=self.n_heads,
                dropout=self.enc_dropout, att_type = self.att_type,
                                    norm_flag=self.norm_flag,
                                    query_cross_time=self.query_cross_time,
                                    att_dropout=self.att_dropout,
                                    return_att=self.return_att,
                                    prob_factor=self.prob_factor,
                                    activation=self.activation_enc) for _ in range(self.num_layers)])
        elif self.att_type == 'difformer':
            self.spatial_agg_list = nn.ModuleList(
            [ DIFFormer_EncoderLayer(self.hid_dim, n_heads=self.n_heads, norm_flag=self.norm_flag,
                                        query_cross_time=self.query_cross_time, dropout=self.enc_dropout, return_att=self.return_att, activation=self.activation_enc) for _ in range(self.num_layers)]
            )
        elif self.att_type=='proxy':
            self.spatial_agg_list = nn.ModuleList([
                EncoderLayer(self.M, self.hid_dim, n_heads=self.n_heads, num_nodes=self.num_nodes, norm_flag=self.norm_flag,
                        query_cross_time=self.query_cross_time,dropout=self.enc_dropout, dynamic_proxy = True,
                        att_dropout=self.att_dropout,return_att=self.return_att,activation=self.activation_enc) for _ in range(self.num_layers)])

        self.time_wise_predictor = time_wise_predictor(
            self.num_nodes, self.input_length, self.predict_length, self.hid_dim, pre_dim=self.pre_dim, num_of_filters=self.d_out, activation=self.activation_dec)

    def forward(self, x):
        if len(x) == 3:
            x, x_time, _ = x
        else:
            x, x_time = x  # x: (B,T,N,C) x_time:(B,T,2)
        B, T, N, _ = x.shape
        assert x.shape[-1] == self.in_dim

        latestX = x[:, -1:, :, :].repeat([1, self.input_length, 1, 1])
        data = self.data_encoding(x, latestX)
        data = self.x_dropout(data)

        x_time_emb = self.getTemb(x_time)
        data = data + x_time_emb

        x_time_lag_emb = self.get_xTimeLagEmb(x_time_emb[:,-1:], x_time_emb)
        data = data + x_time_lag_emb

        x_spa_emb = self.getSemb(data)
        data = data + x_spa_emb

        if self.useTCN:
            data = data.transpose(1, 3)  # (B,T,N,C)->(B,C,N,T)
            if self.tau > 1:
                data = self.padding(data)
            data = self.time_conv(data)
            data = data.transpose(1, 3)  # (B,C,N,T)->(B,T,N,C)
            assert data.shape[1] == self.input_length

        # Graph Convolution Module
        if self.use_graph_conv:
            graph_out = self.graph_conv(data)
            data = data + self.graph_conv_dropout(graph_out)  # skip connection

        skip = data
        A_list = []
        for i in range(self.num_layers):
            data, A = self.spatial_agg_list[i](data)
            A_list.append(A)
        data += skip

        main_output = self.time_wise_predictor(data)


        if self.input_length == self.predict_length:
            main_output += latestX
        else:
            main_output += latestX[:, 0:self.predict_length, :, :]

        if self.return_att:
            return main_output, A_list
        return main_output, None
