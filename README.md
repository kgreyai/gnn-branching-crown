# 基于GNN的神经网络验证器分支策略

本文档说明如何复现"基于图神经网络的验证器分支策略"的完整实验流程，包括环境配置、数据收集、模型训练和对比实验。
- 本项目是西安理工大学计223王嘉安的毕设项目。
- 基于abcrown，改变其分支定界的分支策略(kfsb)，替换为图神经网络来决策。

---

## 目录

1. [环境配置](#1-环境配置)
2. [新增文件说明](#2-新增文件说明)
3. [第一步：收集训练数据](#3-第一步收集训练数据)
4. [第二步：训练GNN模型](#4-第二步训练gnn模型)
5. [第三步：运行对比实验](#5-第三步运行对比实验)
6. [第四步：解析实验结果](#6-第四步解析实验结果)
7. [配置文件说明](#7-配置文件说明)
8. [常见问题](#8-常见问题)

---

## 1. 环境配置

### 1.1 基础环境

本项目基于 alpha-beta-CROWN 原始环境，额外无需安装新依赖。

```bash
# 进入工作目录
cd complete_verifier

# 确认 PyTorch 可用
python -c "import torch; print(torch.__version__)"

# 确认 alpha-beta-CROWN 基础运行正常
python abcrown.py --config exp_configs/beta_crown/mnist_6_100.yaml --start 0 --end 1
```

### 1.2 模型文件准备

确认以下预训练网络权重文件存在：

```bash
ls models/eran/
# 应包含：
# mnist_6_100_nat.pth
# mnist_6_200_nat.pth
# mnist_9_100_nat.pth
```

---

## 2. 新增文件说明

相较于原始 alpha-beta-CROWN，本项目新增/修改了以下文件：

### 新增文件

| 文件 | 说明 |
|------|------|
| `wja_gnn_utils.py` | GNN模型定义（BranchGNN）+ 图特征提取工具 |
| `wja_collect_data.py` | 数据收集工具函数，由 bab.py 钩子调用 |
| `wja_train_gnn.py` | 离线训练脚本 |
| `heuristics/gnn.py` | GnnBranching 分支策略类 |
| `exp_configs/gnn_mnist.yaml` | mnist_6_100 GNN验证配置 |
| `exp_configs/collect_mnist.yaml` | mnist_6_100 kfsb数据收集配置 |
| `exp_configs/gnn_mnist_6_200.yaml` | mnist_6_200 GNN验证配置 |
| `exp_configs/kfsb_mnist_6_200.yaml` | mnist_6_200 kfsb基线配置 |
| `exp_configs/gnn_mnist_9_100.yaml` | mnist_9_100 GNN验证配置 |
| `exp_configs/kfsb_mnist_9_100.yaml` | mnist_9_100 kfsb基线配置 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `bab.py` | 添加 WJA_COLLECT 数据收集钩子（约5行） |
| `arguments.py` | 注册 gnn_model_path、gnn_fallback 配置项；添加 gnn 到分支策略选项 |
| `heuristics/branching_heuristics.py` | 添加 gnn 策略的工厂方法 |

---

## 3. 第一步：收集训练数据

使用 kFSB 策略运行验证，同时记录每步分支决策作为训练数据。

### 3.1 Linux / 远程GPU（推荐）

```bash
cd complete_verifier

# 设置环境变量
export WJA_COLLECT=1
export WJA_SAVE_DIR=./branch_data

# 开始收集（以 MNIST_6_100 为例，运行前100个验证问题）
python abcrown.py \
    --config exp_configs/collect_mnist.yaml \
    --start 0 --end 100

# 取消收集标志（避免后续验证实验误触发）
export WJA_COLLECT=0
```

### 3.2 Windows 本地

```cmd
cd complete_verifier

set WJA_COLLECT=1
set WJA_SAVE_DIR=./branch_data

python abcrown.py --config exp_configs/collect_mnist.yaml --start 0 --end 100

set WJA_COLLECT=0
```

### 3.3 数据收集说明

- 每次分支决策生成一个 `.pt` 文件，保存到 `WJA_SAVE_DIR` 目录
- 文件命名格式：`unknown_step{N}_b{B}.pt`（N=全局步数，B=批次索引）
- 建议至少收集 **4个不同 problem ID**（`--start 0 --end 4`）的数据，约10万+样本
- 可多次运行（不同 `--start/--end` 范围）来扩充数据集，文件会自动累积

### 3.4 验证数据质量

```bash
python -c "
import os, torch
data_dir = './branch_data'
files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
print(f'总样本数: {len(files)}')

# 检查前3个样本
for f in files[:3]:
    d = torch.load(os.path.join(data_dir, f), map_location='cpu')
    N = d['node_features'].shape[0]
    print(f'  {f}: N={N}, label={d[\"label\"]}')
"
```

---

## 4. 第二步：训练GNN模型

### 4.1 基本训练命令

```bash
cd complete_verifier

python wja_train_gnn.py \
    --data_dir ./branch_data \
    --epochs 50 \
    --lr 1e-3 \
    --hidden_dim 64 \
    --num_mp_rounds 2 \
    --max_samples 20000 \
    --output gnn_branch_model.pt
```

### 4.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./branch_data` | 训练数据目录 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-3 | 初始学习率 |
| `--hidden_dim` | 64 | GNN隐层维度 |
| `--num_mp_rounds` | 2 | 消息传递轮数 |
| `--max_samples` | 20000 | 每epoch最大样本数（防内存溢出） |
| `--output` | `gnn_branch_model.pt` | 输出模型路径 |
| `--val_ratio` | 0.2 | 验证集比例 |

### 4.3 预期训练结果

训练过程中每个 epoch 会打印以下指标：

```
Epoch 01/50 | Loss: 5.234 | Top1: 12.3% | Top5: 45.6%  [train]
Epoch 01/50 | Loss: 5.156 | Top1: 14.1% | Top5: 48.2%  [val]
...
Epoch 50/50 | Loss: 2.134 | Top1: 56.4% | Top5: 94.3%  [val]
```

参考性能指标（50个epoch，14万样本）：
- **验证集 Top-1 准确率：≥ 50%**
- **验证集 Top-5 准确率：≥ 90%**

训练完成后模型保存为 `gnn_branch_model.pt`（约214KB）。

---

## 5. 第三步：运行对比实验

确保 `gnn_branch_model.pt` 位于 `complete_verifier/` 目录下。

### 5.1 mnist_6_100 对比实验

在两个终端中分别运行：

```bash
# 终端1：GNN策略
python abcrown.py \
    --config exp_configs/gnn_mnist.yaml \
    --start 0 --end 50 \
    --results_file out_gnn_6_100.pkl

# 终端2：kfsb基线
python abcrown.py \
    --config exp_configs/collect_mnist.yaml \
    --start 0 --end 50 \
    --results_file out_kfsb_6_100.pkl
```

### 5.2 mnist_6_200 对比实验

```bash
# 终端1：GNN策略（零样本迁移）
python abcrown.py \
    --config exp_configs/gnn_mnist_6_200.yaml \
    --start 0 --end 50 \
    --results_file out_gnn_6_200.pkl

# 终端2：kfsb基线
python abcrown.py \
    --config exp_configs/kfsb_mnist_6_200.yaml \
    --start 0 --end 50 \
    --results_file out_kfsb_6_200.pkl
```

### 5.3 mnist_9_100 对比实验

```bash
# 终端1：GNN策略
python abcrown.py \
    --config exp_configs/gnn_mnist_9_100.yaml \
    --start 0 --end 50 \
    --results_file out_gnn_9_100.pkl

# 终端2：kfsb基线
python abcrown.py \
    --config exp_configs/kfsb_mnist_9_100.yaml \
    --start 0 --end 50 \
    --results_file out_kfsb_9_100.pkl
```

---

## 6. 第四步：解析实验结果

实验结果文件为 pickle 格式（`.pkl`），使用以下脚本解析：

### 6.1 单文件解析

```python
import pickle

with open('out_gnn_6_100.pkl', 'rb') as f:
    data = pickle.load(f)

print('结果类别分布:', dict(data['summary']))
# 示例输出: {'safe-incomplete': [0,1,3,...], 'unknown': [2,4,...]}
```

### 6.2 两个结果文件对比

```python
python -c "
import pickle

def parse(fname, label):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    summary = data['summary']
    results = data['results']
    verified = (len(summary.get('safe-incomplete', [])) +
                len(summary.get('safe', [])) +
                len(summary.get('unsat', [])))
    unknown  = len(summary.get('unknown', []))
    total    = len(results)
    times    = [r[1] for r in results]
    safe_times = [r[1] for r in results if r[0] in ('safe-incomplete', 'safe', 'unsat')]
    print(f'[{label}]')
    print(f'  total={total}, verified={verified}, unknown={unknown}, acc={verified/total*100:.1f}%')
    print(f'  avg_time_all={sum(times)/len(times):.2f}s, max_time={max(times):.2f}s')
    if safe_times:
        print(f'  avg_time_safe={sum(safe_times)/len(safe_times):.2f}s')

parse('out_gnn_6_100.pkl', 'GNN  (mnist_6_100)')
parse('out_kfsb_6_100.pkl', 'kfsb (mnist_6_100)')
"
```

### 6.3 多组实验汇总对比

```python
python -c "
import pickle

def parse(fname, label):
    try:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f'[{label}] 文件未找到: {fname}')
        return
    summary = data['summary']
    results = data['results']
    verified = (len(summary.get('safe-incomplete', [])) +
                len(summary.get('safe', [])) +
                len(summary.get('unsat', [])))
    unknown  = len(summary.get('unknown', []))
    total    = len(results)
    times    = [r[1] for r in results]
    safe_times = [r[1] for r in results if r[0] in ('safe-incomplete', 'safe', 'unsat')]
    avg_safe = sum(safe_times)/len(safe_times) if safe_times else float('nan')
    print(f'{label:<30} | verified={verified:>3}/{total} ({verified/total*100:5.1f}%) '
          f'| avg_all={sum(times)/len(times):6.1f}s | avg_safe={avg_safe:6.2f}s')

print(f'{\"策略\":<30} | 验证结果        | 全体均时     | 安全均时')
print('-' * 80)
parse('out_gnn_6_100.pkl',  'GNN   mnist_6_100')
parse('out_kfsb_6_100.pkl', 'kfsb  mnist_6_100')
print()
parse('out_gnn_6_200.pkl',  'GNN   mnist_6_200')
parse('out_kfsb_6_200.pkl', 'kfsb  mnist_6_200')
print()
parse('out_gnn_9_100.pkl',  'GNN   mnist_9_100')
parse('out_kfsb_9_100.pkl', 'kfsb  mnist_9_100')
"
```

---

## 7. 配置文件说明

### 7.1 GNN验证配置关键字段

```yaml
bab:
  branching:
    method: gnn              # 使用GNN分支策略
    gnn_model_path: gnn_branch_model.pt  # 模型文件路径（相对于运行目录）
    gnn_fallback: babsr      # 模型加载失败时的回退策略
    candidates: 3            # top-k候选数（与kfsb保持一致）
```

### 7.2 数据收集配置关键字段

```yaml
bab:
  branching:
    method: kfsb             # 使用kfsb作为专家策略
    candidates: 3
```

### 7.3 模型路径说明

- 默认 `gnn_model_path: gnn_branch_model.pt` 表示在**运行目录**（`complete_verifier/`）下查找
- 若模型文件不存在，会自动打印警告并回退到 babsr 策略，**不会报错退出**

---

## 8. 常见问题

**Q1：收集数据时没有生成 .pt 文件**

检查环境变量是否正确设置：
```bash
echo $WJA_COLLECT   # 应输出 1
echo $WJA_SAVE_DIR  # 应输出保存目录路径
```
Windows下使用 `echo %WJA_COLLECT%`。

---

**Q2：训练时报 EOFError 或文件损坏**

部分 `.pt` 文件可能在中断收集时不完整，训练脚本已内置跳过机制，可忽略警告继续训练。

---

**Q3：验证时报 "模型文件不存在"**

确认 `gnn_branch_model.pt` 位于 `complete_verifier/` 目录下（即运行 `python abcrown.py` 的目录）：
```bash
ls complete_verifier/gnn_branch_model.pt
```

---

**Q4：结果文件用文本编辑器打开是乱码**

结果文件为 Python pickle 二进制格式，必须用 pickle 加载，参见第6节的解析脚本。

---

**Q5：如何只运行10个样本快速测试**

```bash
python abcrown.py --config exp_configs/gnn_mnist.yaml --start 0 --end 10
```

---

**Q6：如何在 Windows 本地复现（无GPU）**

将所有配置文件中的 `device: cuda` 改为 `device: cpu`，其余步骤相同。训练和验证速度会较慢。

---

## 附：文件树结构

```
complete_verifier/
├── wja_gnn_utils.py          # GNN模型 + 特征提取
├── wja_collect_data.py       # 数据收集
├── wja_train_gnn.py          # 训练脚本
├── gnn_branch_model.pt       # 训练好的模型（训练后生成）
├── branch_data/              # 收集的训练数据（收集后生成）
├── heuristics/
│   ├── gnn.py                # GnnBranching 类
│   ├── branching_heuristics.py  # 已添加 gnn 工厂方法
│   └── ...
├── exp_configs/
│   ├── gnn_mnist.yaml        # mnist_6_100 GNN
│   ├── collect_mnist.yaml    # mnist_6_100 kfsb（数据收集 & 基线）
│   ├── gnn_mnist_6_200.yaml  # mnist_6_200 GNN
│   ├── kfsb_mnist_6_200.yaml # mnist_6_200 kfsb
│   ├── gnn_mnist_9_100.yaml  # mnist_9_100 GNN
│   ├── kfsb_mnist_9_100.yaml # mnist_9_100 kfsb
│   └── ...
├── bab.py                    # 已添加 WJA_COLLECT 钩子
├── arguments.py              # 已注册 gnn_model_path、gnn_fallback
└── ...
```
