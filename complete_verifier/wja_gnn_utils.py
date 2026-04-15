"""
wja_gnn_utils.py
================
GNN模型定义 + 图特征提取工具。
用于 heuristics/gnn.py 中的 GnnBranching 类以及 wja_train_gnn.py 训练脚本。

图结构设计：
  - 节点：所有不稳定 ReLU 神经元（mask=1）
  - 节点特征：8维（lb, ub, center, width, neg_ratio, lA_magnitude, layer_idx_norm, intercept_score）
  - 消息传递：层级均值聚合（避免依赖LiRPA内部权重矩阵），2轮前向+反向传播

参考论文：Lu & Kumar, "Neural network branching for neural network verification", ICLR 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# 节点特征提取
# ---------------------------------------------------------------------------

def extract_node_features_single(
    domains: dict,
    batch_idx: int,
    split_activations: dict,
    split_nodes: list,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    从 domains 字典中提取单个批次元素的节点特征。

    Args:
        domains: bab.py 传入的域字典，含 lower_bounds, upper_bounds, lAs, mask
        batch_idx: 批次内的样本索引
        split_activations: net.split_activations，用于找 lA key
        split_nodes: net.split_nodes，有序的 ReLU 层列表

    Returns:
        feats: (N, 8) 节点特征张量，N = 该批次中不稳定神经元总数
        layer_assigns: (N,) int64，每个神经元属于哪一层（0-indexed，按 split_nodes 顺序）
        abs_neuron_indices: (N,) int64，每个神经元在其所在层的绝对索引（flatten后）
        layer_names: List[str]，对应各层的 layer name（与 lower_bounds 的 key 一致）

    Note:
        不稳定神经元由 (lb < 0) & (ub > 0) 判断，而不是 mask。
        mask 在分支过程中可能被修改（已分支的神经元会被清零），
        导致 kfsb 选中的神经元在 mask 中不可见。
    """
    lower_bounds = domains['lower_bounds']
    upper_bounds = domains['upper_bounds']
    lAs_dict = domains.get('lAs', {})

    # 按 split_nodes 确定层的顺序
    ordered_layer_names = []
    for node in split_nodes:
        name = node.name
        if name in lower_bounds:
            ordered_layer_names.append(name)

    total_layers = len(ordered_layer_names)
    if total_layers == 0:
        return (
            torch.zeros(0, 8),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            [],
        )

    all_feats = []
    all_layer_assigns = []
    all_abs_indices = []

    for l_idx, layer_name in enumerate(ordered_layer_names):
        lb = lower_bounds[layer_name][batch_idx].flatten()   # (neurons,)
        ub = upper_bounds[layer_name][batch_idx].flatten()   # (neurons,)

        # 用 (lb < 0) & (ub > 0) 判断不稳定神经元，而非 mask
        unstable = (lb < 0) & (ub > 0)
        if not unstable.any():
            continue

        abs_indices = unstable.nonzero(as_tuple=True)[0]    # (n,) absolute indices
        lb_u = lb[unstable]
        ub_u = ub[unstable]
        n = lb_u.size(0)

        # lA magnitude：来自 CROWN 系数，衡量该神经元对输出边界的贡献
        lA_magnitude = torch.zeros(n, device=lb.device)
        # lA key：从 split_activations 获取对应的激活层名
        try:
            if layer_name in split_activations:
                act_node = split_activations[layer_name][0][0]
                lA_key = act_node.name
                if lA_key in lAs_dict:
                    lA = lAs_dict[lA_key]  # (batch, num_specs, neurons)
                    lA_b = lA[batch_idx]   # (num_specs, neurons)
                    lA_b_flat = lA_b.flatten(0, 0).reshape(lA_b.size(0), -1)
                    # 对 unstable 神经元取绝对值均值
                    lA_b_unstable = lA_b_flat[:, unstable]  # (num_specs, n)
                    lA_magnitude = lA_b_unstable.abs().mean(dim=0)  # (n,)
        except Exception:
            pass  # lA 不可用时退化为 0

        # 计算 8 维特征
        width = (ub_u - lb_u).clamp(min=1e-8)
        center = (lb_u + ub_u) / 2.0
        neg_ratio = (-lb_u.clamp(max=0.0)) / width          # ReLU 松弛斜率 [0,1]
        layer_idx_norm = torch.full((n,), l_idx / max(total_layers - 1, 1),
                                    device=lb.device)
        intercept_score = (width / 4.0) * lA_magnitude       # 综合得分

        feats = torch.stack([
            lb_u.clamp(-10.0, 0.0),          # 0: lower_bound
            ub_u.clamp(0.0, 10.0),            # 1: upper_bound
            center.clamp(-10.0, 10.0),        # 2: center
            width.clamp(0.0, 20.0),           # 3: width
            neg_ratio,                         # 4: neg_ratio [0,1]
            lA_magnitude.clamp(0.0, 10.0),    # 5: lA_magnitude
            layer_idx_norm,                    # 6: layer_idx_norm [0,1]
            intercept_score.clamp(0.0, 10.0), # 7: intercept_score
        ], dim=1)  # (n, 8)

        all_feats.append(feats)
        all_layer_assigns.append(torch.full((n,), l_idx, dtype=torch.long, device=lb.device))
        all_abs_indices.append(abs_indices)

    if len(all_feats) == 0:
        return (
            torch.zeros(0, 8),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            ordered_layer_names,
        )

    return (
        torch.cat(all_feats, dim=0),           # (N, 8)
        torch.cat(all_layer_assigns, dim=0),   # (N,)
        torch.cat(all_abs_indices, dim=0),     # (N,)
        ordered_layer_names,
    )


def build_unstable_index_map(
    domains: dict,
    batch_idx: int,
    split_nodes: list,
) -> Dict[str, torch.Tensor]:
    """
    构建每层不稳定神经元在原始 layer tensor 中的索引，
    用于将 GNN 输出的全局分数还原到各层的分数张量。

    使用 (lb < 0) & (ub > 0) 判断不稳定神经元（与 extract_node_features_single 保持一致）。

    Returns:
        {layer_name: tensor(n_unstable,) of original neuron indices}
    """
    lower_bounds = domains['lower_bounds']
    upper_bounds = domains['upper_bounds']
    result = {}
    for node in split_nodes:
        name = node.name
        if name not in lower_bounds:
            continue
        lb = lower_bounds[name][batch_idx].flatten()
        ub = upper_bounds[name][batch_idx].flatten()
        result[name] = ((lb < 0) & (ub > 0)).nonzero(as_tuple=True)[0]
    return result


# ---------------------------------------------------------------------------
# GNN 模型
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """两层MLP，ReLU激活。"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchGNN(nn.Module):
    """
    基于层级消息传递的 GNN，用于预测分支得分。

    架构：
      1. 节点编码器：Linear(8 → hidden_dim) + ReLU
      2. 层级消息传递（num_mp_rounds 轮）：
         a. 层聚合：每层神经元嵌入取均值 → 层嵌入
         b. 前向消息：上一层嵌入 → MLP → 当前层每个神经元
         c. 反向消息：下一层嵌入 → MLP → 当前层每个神经元
         d. 节点更新：MLP([h_j, fwd_msg_j, bkd_msg_j]) → 新 h_j
      3. 输出头：Linear(hidden_dim → 1)，得到每个神经元的得分

    选择理由（GIN风格 vs GCN/GAT）：
      - 不依赖权重矩阵，只使用层聚合（避免LiRPA内部访问复杂性）
      - 前向+反向双向传播（仿参考论文，捕获全局上下文）
      - GIN 表达能力强于 GCN，比 GAT 实现简单
    """

    def __init__(self, node_feat_dim: int = 8, hidden_dim: int = 64,
                 num_mp_rounds: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_mp_rounds = num_mp_rounds

        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 前向消息 MLP（层嵌入 → 消息）
        self.fwd_msg_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        # 反向消息 MLP
        self.bkd_msg_mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
        # 节点更新 MLP（输入：[h, fwd_msg, bkd_msg]）
        self.update_mlp = MLP(hidden_dim * 3, hidden_dim, hidden_dim)

        # 输出头
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_feats: torch.Tensor,       # (N, 8)
        layer_assigns: torch.Tensor,    # (N,) int64
        num_layers: int,
    ) -> torch.Tensor:
        """
        Returns:
            scores: (N,) 每个不稳定神经元的得分（未归一化）
        """
        device = node_feats.device
        N = node_feats.size(0)

        if N == 0:
            return torch.zeros(0, device=device)

        # 1. 编码节点特征
        h = self.node_encoder(node_feats)  # (N, hidden_dim)

        # 2. 多轮消息传递
        for _ in range(self.num_mp_rounds):
            # a. 层聚合：每层神经元嵌入取均值 → (num_layers, hidden_dim)
            layer_h = torch.zeros(num_layers, self.hidden_dim, device=device)
            layer_count = torch.zeros(num_layers, device=device)
            layer_h.scatter_add_(
                0,
                layer_assigns.unsqueeze(1).expand(-1, self.hidden_dim),
                h,
            )
            layer_count.scatter_add_(
                0,
                layer_assigns,
                torch.ones(N, device=device),
            )
            # 避免除零
            layer_count = layer_count.clamp(min=1.0)
            layer_h = layer_h / layer_count.unsqueeze(1)  # (num_layers, hidden_dim)

            # b/c. 计算前向和反向消息
            fwd_msgs = self.fwd_msg_mlp(layer_h)  # (num_layers, hidden_dim)
            bkd_msgs = self.bkd_msg_mlp(layer_h)  # (num_layers, hidden_dim)

            # 每个神经元的前向消息来自上一层，反向消息来自下一层
            # layer l 的前向消息 = fwd_msgs[l-1]（若 l=0 则为 0）
            # layer l 的反向消息 = bkd_msgs[l+1]（若 l=L-1 则为 0）
            fwd_layer = torch.zeros_like(layer_h)  # (num_layers, hidden_dim)
            bkd_layer = torch.zeros_like(layer_h)
            if num_layers > 1:
                fwd_layer[1:] = fwd_msgs[:-1]   # layer l 从 layer l-1 获取前向消息
                bkd_layer[:-1] = bkd_msgs[1:]   # layer l 从 layer l+1 获取反向消息

            # d. 每个神经元获取其层的消息
            fwd_per_neuron = fwd_layer[layer_assigns]   # (N, hidden_dim)
            bkd_per_neuron = bkd_layer[layer_assigns]   # (N, hidden_dim)

            # e. 节点更新
            h = self.update_mlp(
                torch.cat([h, fwd_per_neuron, bkd_per_neuron], dim=1)
            )  # (N, hidden_dim)

        # 3. 输出得分
        scores = self.score_head(h).squeeze(-1)  # (N,)
        return scores


# ---------------------------------------------------------------------------
# 模型保存 / 加载
# ---------------------------------------------------------------------------

def save_model(model: BranchGNN, path: str, extra: Optional[dict] = None):
    """保存模型 checkpoint。"""
    ckpt = {
        'state_dict': model.state_dict(),
        'node_feat_dim': model.node_encoder[0].in_features,  # Linear(node_feat_dim, hidden_dim)
        'hidden_dim': model.hidden_dim,
        'num_mp_rounds': model.num_mp_rounds,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_model(path: str, device: str = 'cpu') -> BranchGNN:
    """从 checkpoint 加载模型。"""
    ckpt = torch.load(path, map_location=device)
    # 从 state_dict 直接推断 node_feat_dim，避免 checkpoint 里存错的情况
    state = ckpt['state_dict']
    node_feat_dim = state['node_encoder.0.weight'].shape[1]
    model = BranchGNN(
        node_feat_dim=node_feat_dim,
        hidden_dim=ckpt.get('hidden_dim', 64),
        num_mp_rounds=ckpt.get('num_mp_rounds', 2),
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model
