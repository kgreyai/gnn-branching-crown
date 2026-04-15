"""
wja_collect_data.py
===================
数据收集工具。在 kfsb 分支策略运行时，记录每次分支决策的状态和标签，
用于后续 GNN 模型的离线训练（模仿学习 / imitation learning）。

使用方式：
  1. 在 bab.py 的 split_domain() 中调用 collect_branch_state()
  2. 运行结束后用 check_data() 检查数据集质量

数据格式（每个 .pt 文件）：
  {
      'node_features':   Tensor(N, 8),   # 所有不稳定神经元的特征
      'layer_assigns':   Tensor(N,) int64,  # 每个神经元属于哪层（0-indexed）
      'chosen_global_idx': int,           # 被选神经元的全局索引（在 N 个中）
      'total_layers':    int,
      'problem_id':      str,             # 验证问题标识
      'batch_idx':       int,
      'step_id':         int,             # 当前验证问题的第几次分支
  }
"""

import os
import glob
import torch
import argparse
from typing import Optional

from wja_gnn_utils import extract_node_features_single


# ---------------------------------------------------------------------------
# 全局状态（跨调用共享）
# ---------------------------------------------------------------------------

_collect_state = {
    'initialized': False,
    'save_dir': './branch_data',
    'problem_id': 'unknown',
    'step_counter': 0,
    'sample_counter': 0,
}


def _auto_init():
    """首次调用时从环境变量自动初始化（无需手动调用 init_collector）。"""
    if not _collect_state['initialized']:
        save_dir = os.environ.get('WJA_SAVE_DIR', './branch_data').strip()
        os.makedirs(save_dir, exist_ok=True)
        _collect_state['save_dir'] = save_dir
        _collect_state['initialized'] = True
        print(f'[wja_collect] save_dir = {os.path.abspath(save_dir)}')


def init_collector(save_dir: str, problem_id: str = 'unknown'):
    """可选：手动初始化，指定保存目录和问题ID。"""
    os.makedirs(save_dir, exist_ok=True)
    _collect_state['initialized'] = True
    _collect_state['save_dir'] = save_dir
    _collect_state['problem_id'] = str(problem_id)
    _collect_state['step_counter'] = 0


def reset_collector():
    """重置计数器（每个验证问题开始时可调用）。"""
    _collect_state['step_counter'] = 0


def collect_branch_state(
    domains: dict,
    branching_decision: list,
    split_depth: int,
    net,
    batch_idx: int = 0,
):
    """
    在 bab.py 的 split_domain() 中调用，保存一次分支状态和标签。

    Args:
        domains: 当前域字典（lower_bounds, upper_bounds, lAs, mask）
        branching_decision: get_branching_decisions 的返回值第一项，
                            格式为 List[(layer_idx, neuron_idx)]
        split_depth: 本次分支深度（通常为1）
        net: LiRPANet 对象（提供 split_nodes, split_activations）
        batch_idx: 批次内的样本索引（默认0）
    """
    _auto_init()

    # One-time debug: print key structures on first call
    if _collect_state.get('_debug_printed', False) is False:
        _collect_state['_debug_printed'] = True
        lb_keys = list(domains['lower_bounds'].keys())
        sn_names = [n.name for n in net.split_nodes]
        sa_keys = list(net.split_activations.keys())
        print(f'[wja debug] lower_bounds keys (first 5): {lb_keys[:5]}')
        print(f'[wja debug] split_nodes names: {sn_names}')
        print(f'[wja debug] split_activations keys: {sa_keys[:5]}')
        print(f'[wja debug] mask keys (first 5): {list(domains["mask"].keys())[:5]}')

    try:
        # 提取节点特征（返回4个值，abs_neuron_indices 是每个神经元在其层内的绝对索引）
        feats, layer_assigns, abs_neuron_indices, layer_names = extract_node_features_single(
            domains, batch_idx,
            net.split_activations,
            net.split_nodes,
        )

        N = feats.size(0)
        if N < 2:
            print(f'[wja R1] N={N} < 2, skip. layer_names={layer_names}')
            return

        dec_offset = batch_idx * split_depth
        if dec_offset >= len(branching_decision):
            print(f'[wja R2] dec_offset={dec_offset} >= len(dec)={len(branching_decision)}')
            return
        dec_layer_idx, dec_neuron_idx = branching_decision[dec_offset]
        # GPU模式下可能是tensor，统一转为Python int
        if hasattr(dec_layer_idx, 'item'):
            dec_layer_idx = dec_layer_idx.item()
        if hasattr(dec_neuron_idx, 'item'):
            dec_neuron_idx = dec_neuron_idx.item()

        if dec_layer_idx < 0 or dec_layer_idx >= len(net.split_nodes):
            print(f'[wja R3] dec_layer_idx={dec_layer_idx} out of range {len(net.split_nodes)}')
            return
        chosen_layer_name = net.split_nodes[dec_layer_idx].name

        if chosen_layer_name not in layer_names:
            if _collect_state.get('_r4_count', 0) < 2:
                _collect_state['_r4_count'] = _collect_state.get('_r4_count', 0) + 1
                print(f'[wja R4] chosen={chosen_layer_name} not in layer_names={layer_names}')
                print(f'[wja R4] ALL lower_bounds keys: {list(domains["lower_bounds"].keys())}')
            return
        chosen_l_idx = layer_names.index(chosen_layer_name)

        # 在所有不稳定神经元中找到被选神经元的全局位置
        # 通过 (layer_assigns == chosen_l_idx) & (abs_neuron_indices == dec_neuron_idx)
        matches = (layer_assigns == chosen_l_idx) & (abs_neuron_indices == dec_neuron_idx)
        match_positions = matches.nonzero(as_tuple=True)[0]
        if len(match_positions) == 0:
            # 被选神经元不在不稳定列表中（可能 lb>=0 或 ub<=0）
            layer_abs = abs_neuron_indices[layer_assigns == chosen_l_idx]
            print(f'[wja R6] neuron (layer={chosen_l_idx}, abs_idx={dec_neuron_idx}) '
                  f'not in unstable neurons {layer_abs[:10].tolist()}')
            return
        global_idx = match_positions[0].item()

        # 保存
        sample = {
            'node_features':     feats.cpu(),
            'layer_assigns':     layer_assigns.cpu(),
            'chosen_global_idx': global_idx,
            'total_layers':      len(layer_names),
            'problem_id':        _collect_state['problem_id'],
            'batch_idx':         batch_idx,
            'step_id':           _collect_state['step_counter'],
        }

        step = _collect_state['step_counter']
        pid = _collect_state['problem_id']
        filename = f"{pid}_step{step:05d}_b{batch_idx}.pt"
        filepath = os.path.join(_collect_state['save_dir'], filename)
        torch.save(sample, filepath)

        _collect_state['step_counter'] += 1
        _collect_state['sample_counter'] += 1

    except Exception as e:
        import traceback
        print(f'[wja collect ERROR] {e}')
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 数据集检查工具
# ---------------------------------------------------------------------------

def check_data(data_dir: str, max_files: int = 100):
    """
    检查收集的数据集质量，打印统计信息。

    Args:
        data_dir: 数据目录
        max_files: 最多检查的文件数
    """
    files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
    if len(files) == 0:
        print(f"[check_data] 未找到 .pt 文件：{data_dir}")
        return

    print(f"[check_data] 找到 {len(files)} 个文件，检查前 {min(max_files, len(files))} 个")

    n_neurons_list = []
    n_layers_list = []
    chosen_idx_list = []
    valid = 0
    invalid = 0

    for fpath in files[:max_files]:
        try:
            sample = torch.load(fpath, map_location='cpu')
            N = sample['node_features'].size(0)
            L = sample['total_layers']
            c = sample['chosen_global_idx']

            if N < 2 or c < 0 or c >= N:
                invalid += 1
                continue

            n_neurons_list.append(N)
            n_layers_list.append(L)
            chosen_idx_list.append(c)
            valid += 1
        except Exception as e:
            invalid += 1

    print(f"  有效样本：{valid}，无效样本：{invalid}")
    if valid > 0:
        n_t = torch.tensor(n_neurons_list, dtype=torch.float)
        l_t = torch.tensor(n_layers_list, dtype=torch.float)
        c_t = torch.tensor(chosen_idx_list, dtype=torch.float)
        c_rel = c_t / n_t  # 被选神经元的相对位置
        print(f"  不稳定神经元数：min={n_t.min():.0f}, max={n_t.max():.0f}, "
              f"mean={n_t.mean():.1f}")
        print(f"  层数：min={l_t.min():.0f}, max={l_t.max():.0f}, "
              f"mean={l_t.mean():.1f}")
        print(f"  chosen_idx 相对位置：min={c_rel.min():.3f}, "
              f"max={c_rel.max():.3f}, mean={c_rel.mean():.3f}")

        # 特征统计
        sample = torch.load(files[0], map_location='cpu')
        feats = sample['node_features']
        print(f"  节点特征 shape: {feats.shape}")
        print(f"  特征各维度 mean: {feats.mean(0).numpy().round(4)}")
        print(f"  特征各维度 std:  {feats.std(0).numpy().round(4)}")


# ---------------------------------------------------------------------------
# CLI 入口（用于独立检查数据集）
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='检查GNN分支数据集')
    parser.add_argument('--data_dir', type=str, default='./branch_data',
                        help='数据目录')
    parser.add_argument('--max_files', type=int, default=500,
                        help='最多检查文件数')
    args = parser.parse_args()
    check_data(args.data_dir, args.max_files)
