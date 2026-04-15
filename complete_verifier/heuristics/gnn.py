"""
heuristics/gnn.py
=================
基于 GNN 的神经元分支策略，继承自 NeuronBranchingHeuristic。

使用方式：在 YAML 配置中设置
  bab:
    branching:
      method: gnn
      gnn:
        model_path: gnn_branch_model.pt
        fallback: babsr   # 可选回退策略

实现说明：
  - 每个分支步骤中，提取所有不稳定神经元的图特征，运行 GNN 前向推理，
    将输出得分还原为各层分数张量，交由基类完成 top-k 选择和格式化。
  - 若 GNN 推理失败（模型路径不存在、异常等），自动回退到 babsr 策略。
  - 支持批次处理：对每个批次元素独立运行 GNN，合并结果。
"""

import os
import torch

from heuristics.base import NeuronBranchingHeuristic
from heuristics.babsr import BabsrBranching


class GnnBranching(NeuronBranchingHeuristic):
    """
    GNN 分支策略。

    继承 NeuronBranchingHeuristic，只需实现 compute_neuron_scores()。
    基类负责 top-k 选择（find_topk_scores）和格式化（format_decisions）。
    """

    def __init__(self, net, model_path: str = 'gnn_branch_model.pt',
                 fallback: str = 'babsr'):
        super().__init__(net)
        self.model_path = model_path
        self.gnn_model = None
        self._load_error = None
        self._fallback_branching = None

        # 尝试加载模型
        self._try_load_model()

        # 初始化回退策略
        if fallback == 'babsr':
            self._fallback_branching = BabsrBranching(net)
        # 可根据需要添加其他回退策略

    def _try_load_model(self):
        """尝试加载 GNN 模型，失败时记录错误并使用回退。"""
        if not os.path.exists(self.model_path):
            self._load_error = f"模型文件不存在: {self.model_path}"
            print(f"[GnnBranching] 警告: {self._load_error}，将使用回退策略")
            return
        try:
            from wja_gnn_utils import load_model
            self.gnn_model = load_model(self.model_path, device='cpu')
            self.gnn_model.eval()
            print(f"[GnnBranching] 模型加载成功: {self.model_path}")
        except Exception as e:
            self._load_error = str(e)
            print(f"[GnnBranching] 模型加载失败: {e}，将使用回退策略")

    def get_branching_decisions(self, domains, split_depth=1, **kwargs):
        """模型不可用时直接代理到 fallback 的完整决策流程。"""
        if self.gnn_model is None:
            if self._fallback_branching is not None:
                return self._fallback_branching.get_branching_decisions(
                    domains, split_depth, **kwargs)
            raise RuntimeError(f"GNN 模型不可用且无回退策略: {self._load_error}")
        return super().get_branching_decisions(domains, split_depth, **kwargs)

    def compute_neuron_scores(self, domains, **kwargs):
        """
        计算每个不稳定神经元的分支得分。

        Returns:
            scores: {layer_name: Tensor(batch, neurons)} 分支得分字典
        """
        try:
            return self._gnn_scores(domains)
        except Exception as e:
            if self._fallback_branching is not None:
                return self._fallback_branching.get_branching_decisions(
                    domains, **kwargs)
            raise

    def _gnn_scores(self, domains):
        """核心：运行 GNN 推理并返回各层分数字典。"""
        from wja_gnn_utils import extract_node_features_single

        lower_bounds = domains['lower_bounds']
        batch_size = next(iter(lower_bounds.values())).size(0)
        device = next(iter(lower_bounds.values())).device

        # 初始化各层分数为极小值，防止未被GNN评分的神经元（score=0）意外胜出
        all_scores = {}
        for layer_name, lb in self.layer_iterator(lower_bounds):
            all_scores[layer_name] = torch.full_like(lb.flatten(1), fill_value=-1e9)

        # 逐批次元素运行 GNN
        for b in range(batch_size):
            feats, layer_assigns, abs_neuron_indices, layer_names = extract_node_features_single(
                domains, b,
                self.net.split_activations,
                self.net.split_nodes,
            )

            N = feats.size(0)
            if N < 2:
                # 不稳定神经元太少，跳过（保持 0 分）
                continue

            # GNN 推理（CPU 上运行，避免设备不匹配）
            with torch.no_grad():
                scores_per_neuron = self.gnn_model(
                    feats.cpu(),
                    layer_assigns.cpu(),
                    len(layer_names),
                ).cpu()  # (N,)

            # 将全局分数还原到各层的分数张量
            # abs_neuron_indices[i] 是第 i 个不稳定神经元在其所在层的绝对索引

            global_offset = 0
            for l_idx, layer_name in enumerate(layer_names):
                if layer_name not in all_scores:
                    n_this_layer = (layer_assigns == l_idx).sum().item()
                    global_offset += n_this_layer
                    continue

                # 该层的不稳定神经元数量
                this_layer_mask = (layer_assigns == l_idx)
                n_this_layer = this_layer_mask.sum().item()

                if n_this_layer == 0:
                    continue

                # 该层的 GNN 分数及其原始绝对索引
                layer_scores_gnn = scores_per_neuron[global_offset: global_offset + n_this_layer]
                orig_indices = abs_neuron_indices[this_layer_mask].cpu()
                global_offset += n_this_layer

                n_neurons_total = all_scores[layer_name].size(1)
                for i in range(len(orig_indices)):
                    orig_idx = orig_indices[i].item()
                    if orig_idx < n_neurons_total:
                        all_scores[layer_name][b, orig_idx] = (
                            layer_scores_gnn[i].to(device)
                        )

        return all_scores
