import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphConv(nn.Module):
    """
    自适应图卷积层：学习节点间的邻接矩阵，捕捉路网空间依赖。
    
    核心思路：
    - 为每个节点学习两个嵌入向量（node_emb1, node_emb2）
    - 自适应邻接矩阵 A_adaptive = softmax(relu(E1 @ E2^T))
    - 支持融合静态先验图（如 Pearson 相关系数图）
    
    Args:
        hid_dim: 隐藏维度
        num_nodes: 节点数 N
        use_adaptive: 是否使用自适应图学习
        use_static: 是否使用静态先验图
        static_adj: 静态邻接矩阵 (N, N)，可选
        fusion_type: 融合方式，'gate'（门控）或 'add'（相加）
    """
    def __init__(self, hid_dim, num_nodes, use_adaptive=True, use_static=False,
                 static_adj=None, fusion_type='gate'):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.use_adaptive = use_adaptive
        self.use_static = use_static
        self.fusion_type = fusion_type

        if use_adaptive:
            # 可学习的节点嵌入，用于构建自适应邻接矩阵
            self.node_emb1 = nn.Parameter(torch.randn(num_nodes, hid_dim))
            self.node_emb2 = nn.Parameter(torch.randn(num_nodes, hid_dim))
            nn.init.xavier_uniform_(self.node_emb1)
            nn.init.xavier_uniform_(self.node_emb2)

        if use_static and static_adj is not None:
            # 注册静态邻接矩阵为 buffer（不参与训练，但会随模型保存/加载）
            self.register_buffer('static_adj', static_adj)
            # 预计算归一化的静态图（避免每次 forward 都重新归一化）
            self.register_buffer('static_adj_norm', self._normalize_adj(static_adj))

        if use_adaptive and use_static:
            # 门控融合参数
            if fusion_type == 'gate':
                self.gate = nn.Parameter(torch.tensor(0.5))

        self.W = nn.Linear(hid_dim, hid_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def _normalize_adj(self, A):
        """
        归一化邻接矩阵 A。

        对于 Pearson 相关系数图（值在 [-1, 1]）：
        1. 只保留正相关（ReLU）
        2. softmax 归一化（每行和为 1）

        Args:
            A: (N, N) 邻接矩阵
        Returns:
            A_norm: (N, N) 归一化后的邻接矩阵
        """
        # 只保留正相关（负相关在交通数据中无意义）
        A = F.relu(A)
        # softmax 归一化（每行和为 1）
        A_norm = F.softmax(A, dim=1)
        return A_norm

    def forward(self, x, static_adj=None):
        """
        Args:
            x: (B, T, N, C) 输入张量
            static_adj: (N, N) 静态邻接矩阵（可选，覆盖 self.static_adj）
        Returns:
            output: (B, T, N, C)
        """
        B, T, N, C = x.shape

        # 计算自适应邻接矩阵
        if self.use_adaptive:
            A_adaptive = torch.relu(torch.mm(self.node_emb1, self.node_emb2.T))
            A_adaptive = F.softmax(A_adaptive, dim=1)

        # 获取静态邻接矩阵（优先级：参数 > buffer）
        if static_adj is not None:
            A_static = self._normalize_adj(static_adj)
        elif self.use_static and hasattr(self, 'static_adj_norm'):
            A_static = self.static_adj_norm
        else:
            A_static = None

        # 融合自适应图和静态图
        if self.use_adaptive and A_static is not None:
            if self.fusion_type == 'gate':
                A = self.gate * A_adaptive + (1 - self.gate) * A_static
                # 重新归一化（确保每行和为 1）
                A = F.softmax(A, dim=1)
            elif self.fusion_type == 'add':
                A = A_adaptive + A_static
                A = F.softmax(A, dim=1)
        elif self.use_adaptive:
            A = A_adaptive
        elif A_static is not None:
            A = A_static
        else:
            raise ValueError("Either use_adaptive or use_static must be True")

        # 图卷积：A @ X
        x_reshaped = x.reshape(B * T, N, C)
        output = torch.einsum('nm,bmc->bnc', A, x_reshaped)
        output = self.W(output)
        output = self.activation(output)
        output = self.dropout(output)
        output = output.reshape(B, T, N, C)

        return output

    def get_adaptive_adj(self):
        """返回学习到的自适应邻接矩阵，用于可视化/分析"""
        if self.use_adaptive:
            A_adaptive = torch.relu(torch.mm(self.node_emb1, self.node_emb2.T))
            A_adaptive = F.softmax(A_adaptive, dim=1)
            return A_adaptive
        return None


class MultiScaleGraphConv(nn.Module):
    """
    多尺度图卷积：通过 k-hop 聚合捕捉多尺度空间依赖。
    
    对邻接矩阵 A 做幂运算得到 k-hop 邻接矩阵：
    - A^1: 直接邻居
    - A^2: 2-hop 邻居
    - ...
    
    Args:
        hid_dim: 隐藏维度
        num_nodes: 节点数
        k_hops: 最大跳数（如 [1, 2] 表示 1-hop 和 2-hop）
        use_adaptive: 是否使用自适应图学习
        static_adj: 静态邻接矩阵 (N, N)，可选（作为基础邻接矩阵）
    """
    def __init__(self, hid_dim, num_nodes, k_hops=[1, 2], use_adaptive=True, static_adj=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.k_hops = k_hops
        self.use_adaptive = use_adaptive

        if use_adaptive:
            self.node_emb1 = nn.Parameter(torch.randn(num_nodes, hid_dim))
            self.node_emb2 = nn.Parameter(torch.randn(num_nodes, hid_dim))
            nn.init.xavier_uniform_(self.node_emb1)
            nn.init.xavier_uniform_(self.node_emb2)

        # 如果使用静态图作为基础邻接矩阵
        if static_adj is not None:
            self.register_buffer('static_adj', static_adj)
            self.use_static_base = True
        else:
            self.use_static_base = False

        # 每个 hop 一个独立的线性变换
        self.W_list = nn.ModuleList([
            nn.Linear(hid_dim, hid_dim) for _ in k_hops
        ])
        self.fusion = nn.Linear(hid_dim * len(k_hops), hid_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, C)
        Returns:
            output: (B, T, N, C)
        """
        B, T, N, C = x.shape

        # 计算基础邻接矩阵
        if self.use_adaptive and not self.use_static_base:
            A_base = torch.relu(torch.mm(self.node_emb1, self.node_emb2.T))
            A_base = F.softmax(A_base, dim=1)
        elif self.use_static_base:
            A_base = self.static_adj
        elif self.use_adaptive:
            # 融合自适应图和静态图作为基础邻接矩阵
            A_adaptive = torch.relu(torch.mm(self.node_emb1, self.node_emb2.T))
            A_adaptive = F.softmax(A_adaptive, dim=1)
            A_base = 0.5 * A_adaptive + 0.5 * self.static_adj
        else:
            raise ValueError("Either use_adaptive or static_adj must be provided")

        x_reshaped = x.reshape(B * T, N, C)
        outputs = []

        for i, k in enumerate(self.k_hops):
            # 计算 k-hop 邻接矩阵：A^k
            A_k = torch.matrix_power(A_base, k)
            # 归一化（确保每行和为 1）
            A_k = F.normalize(A_k, p=1, dim=1)

            # 图卷积
            out = torch.einsum('nm,bmc->bnc', A_k, x_reshaped)
            out = self.W_list[i](out)
            outputs.append(out)

        # 融合多尺度特征
        outputs = torch.cat(outputs, dim=-1)  # (B*T, N, C*len(k_hops))
        output = self.fusion(outputs)  # (B*T, N, C)
        output = self.activation(output)
        output = self.dropout(output)
        output = output.reshape(B, T, N, C)

        return output
