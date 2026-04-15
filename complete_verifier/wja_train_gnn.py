"""
wja_train_gnn.py
================
GNN 分支策略的离线训练脚本。

训练方法：模仿学习（Imitation Learning）
  - 目标：训练 GNN 预测 kfsb 的分支决策
  - 损失：交叉熵（在所有不稳定神经元上做多分类）
  - 评估：top-1, top-5, top-10 准确率

使用方式：
  # 基本训练
  python wja_train_gnn.py --data_dir ./branch_data --epochs 50

  # 指定输出路径和超参
  python wja_train_gnn.py --data_dir ./branch_data --epochs 100 \\
      --hidden_dim 128 --lr 1e-3 --output gnn_branch_model.pt
"""

import os
import glob
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    # tqdm 未安装时的简单替代
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable
            self._desc = kwargs.get('desc', '')
            self._total = kwargs.get('total', None)
            self._n = 0
        def __iter__(self):
            for x in self._it:
                self._n += 1
                if self._n % 500 == 0:
                    t = self._total or '?'
                    print(f'  {self._desc} {self._n}/{t}', flush=True)
                yield x
        def set_postfix(self, **kwargs):
            pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): self._n += n

from wja_gnn_utils import BranchGNN, save_model


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

class BranchDataset(Dataset):
    """每个样本对应一次分支决策。"""

    def __init__(self, file_list: List[str], cache_limit: int = 50000):
        self.files = file_list
        self._cache = {}
        self._cache_limit = cache_limit

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        try:
            sample = torch.load(self.files[idx], map_location='cpu', weights_only=False)
            result = (
                sample['node_features'],        # (N, 8)
                sample['layer_assigns'],        # (N,) int64
                int(sample['total_layers']),
                int(sample['chosen_global_idx']),
            )
        except Exception:
            return None  # 损坏文件，调用方跳过
        if len(self._cache) < self._cache_limit:
            self._cache[idx] = result
        return result


def collate_fn(batch):
    """
    自定义 collate：每个样本神经元数不同，不能直接 stack。
    返回列表，逐样本处理。
    """
    return batch  # List of (feats, layer_assigns, total_layers, chosen_idx)


def load_dataset(data_dir: str, val_ratio: float = 0.1, seed: int = 42,
                 max_samples: int = 0):
    """加载并分割数据集。max_samples=0 表示使用全部文件。"""
    files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
    if len(files) == 0:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 .pt 文件")

    print(f"[数据集] 找到 {len(files)} 个文件", end='')

    random.seed(seed)
    random.shuffle(files)

    # 子采样（避免全量扫描）
    if max_samples > 0 and len(files) > max_samples:
        files = files[:max_samples]
        print(f"，随机抽取 {max_samples} 个")
    else:
        print()

    n_val = max(1, int(len(files) * val_ratio))
    val_files = files[:n_val]
    train_files = files[n_val:]

    print(f"[数据集] 训练: {len(train_files)}，验证: {len(val_files)}")
    return BranchDataset(train_files), BranchDataset(val_files)


# ---------------------------------------------------------------------------
# 训练 / 验证
# ---------------------------------------------------------------------------

def compute_topk_acc(logits: torch.Tensor, label: int, ks: Tuple[int, ...] = (1, 5, 10)):
    """计算 top-k 准确率（0 或 1）。"""
    N = logits.size(0)
    results = {}
    for k in ks:
        k_eff = min(k, N)
        topk_indices = logits.topk(k_eff).indices
        results[k] = int(label in topk_indices.tolist())
    return results


def train_epoch(model, train_dataset, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    acc_counts = {1: 0, 5: 0, 10: 0}
    n_samples = 0

    indices = list(range(len(train_dataset)))
    random.shuffle(indices)

    bar = tqdm(indices, desc=f'Epoch {epoch:3d} [train]', ncols=90, leave=False)
    for idx in bar:
        item = train_dataset[idx]
        if item is None:
            continue
        feats, layer_assigns, total_layers, chosen_idx = item
        feats = feats.to(device)
        layer_assigns = layer_assigns.to(device)
        N = feats.size(0)

        if N < 2:
            continue

        logits = model(feats, layer_assigns, total_layers)
        label_tensor = torch.tensor([chosen_idx], dtype=torch.long, device=device)
        loss = F.cross_entropy(logits.unsqueeze(0), label_tensor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        accs = compute_topk_acc(logits.detach().cpu(), chosen_idx)
        for k in acc_counts:
            acc_counts[k] += accs.get(k, 0)
        n_samples += 1

        if n_samples % 200 == 0:
            bar.set_postfix(loss=f'{total_loss/n_samples:.4f}',
                            top1=f'{acc_counts[1]/n_samples:.3f}')

    if n_samples == 0:
        return 0.0, {k: 0.0 for k in acc_counts}

    avg_loss = total_loss / n_samples
    avg_accs = {k: acc_counts[k] / n_samples for k in acc_counts}
    return avg_loss, avg_accs


@torch.no_grad()
def eval_epoch(model, val_dataset, device, epoch):
    model.eval()
    total_loss = 0.0
    acc_counts = {1: 0, 5: 0, 10: 0}
    n_samples = 0

    bar = tqdm(range(len(val_dataset)), desc=f'Epoch {epoch:3d} [ val ]', ncols=90, leave=False)
    for idx in bar:
        item = val_dataset[idx]
        if item is None:
            continue
        feats, layer_assigns, total_layers, chosen_idx = item
        feats = feats.to(device)
        layer_assigns = layer_assigns.to(device)
        N = feats.size(0)

        if N < 2:
            continue

        logits = model(feats, layer_assigns, total_layers)
        label_tensor = torch.tensor([chosen_idx], dtype=torch.long, device=device)
        loss = F.cross_entropy(logits.unsqueeze(0), label_tensor)

        total_loss += loss.item()
        accs = compute_topk_acc(logits.cpu(), chosen_idx)
        for k in acc_counts:
            acc_counts[k] += accs.get(k, 0)
        n_samples += 1

    if n_samples == 0:
        return 0.0, {k: 0.0 for k in acc_counts}

    avg_loss = total_loss / n_samples
    avg_accs = {k: acc_counts[k] / n_samples for k in acc_counts}
    return avg_loss, avg_accs


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[训练] 使用设备: {device}")

    # 加载数据集
    train_dataset, val_dataset = load_dataset(
        args.data_dir, args.val_ratio, args.seed, args.max_samples)

    # 构建模型
    model = BranchGNN(
        node_feat_dim=8,
        hidden_dim=args.hidden_dim,
        num_mp_rounds=args.mp_rounds,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 参数量: {n_params:,}，hidden_dim={args.hidden_dim}，mp_rounds={args.mp_rounds}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    best_val_loss = float('inf')
    best_val_top1 = 0.0
    best_epoch = 0

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'T-Top1':>8} {'T-Top5':>8} "
          f"{'ValLoss':>10} {'V-Top1':>8} {'V-Top5':>8} {'LR':>8}")
    print('-' * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accs = train_epoch(model, train_dataset, optimizer, device, epoch)
        val_loss, val_accs = eval_epoch(model, val_dataset, device, epoch)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:>6} {train_loss:>10.4f} {train_accs[1]:>8.3f} {train_accs[5]:>8.3f} "
              f"{val_loss:>10.4f} {val_accs[1]:>8.3f} {val_accs[5]:>8.3f} {current_lr:>8.6f}")

        # 保存最佳模型（按验证 top-1 acc）
        if val_accs[1] > best_val_top1 or (val_accs[1] == best_val_top1 and val_loss < best_val_loss):
            best_val_top1 = val_accs[1]
            best_val_loss = val_loss
            best_epoch = epoch
            save_model(model, args.output, extra={
                'epoch': epoch,
                'val_loss': val_loss,
                'val_top1': val_accs[1],
                'val_top5': val_accs[5],
            })

    print(f"\n[训练完成] 最佳验证 top-1 acc: {best_val_top1:.3f}（第 {best_epoch} 轮）")
    print(f"[模型保存] {args.output}")

    # 也保存最后一轮
    last_path = args.output.replace('.pt', '_last.pt')
    save_model(model, last_path, extra={'epoch': args.epochs})
    print(f"[最后轮次] {last_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练GNN分支策略模型')
    parser.add_argument('--data_dir', type=str, default='./branch_data',
                        help='数据目录（包含 .pt 文件）')
    parser.add_argument('--output', type=str, default='gnn_branch_model.pt',
                        help='模型输出路径')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='GNN隐藏层维度')
    parser.add_argument('--mp_rounds', type=int, default=2,
                        help='消息传递轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--max_samples', type=int, default=20000,
                        help='最多使用的样本数（0=全部，默认20000）')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')
    args = parser.parse_args()
    train(args)
