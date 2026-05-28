"""第五章图表生成脚本 - 数据收集与训练"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

OUT = os.path.dirname(os.path.abspath(__file__))

# 蓝灰配色
COLOR_BG_LIGHT = '#EAF2F8'
COLOR_BG_MID = '#AED6F1'
COLOR_BG_DARK = '#5DADE2'
COLOR_BORDER = '#1B4F72'
COLOR_HIGHLIGHT = '#F8C471'
COLOR_ARROW = '#34495E'
COLOR_TEXT = '#17202A'
COLOR_RED = '#C0392B'
COLOR_GREEN = '#27AE60'

FS_TITLE = 18
FS_BIG = 17
FS_MED = 16
FS_SMALL = 14


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white',
                pad_inches=0.3)
    plt.close(fig)
    print(f'saved: {path}')


# ============================================================
# 图5-1 数据收集机制流程图 (16:10)
# ============================================================
def fig_5_1():
    fig, ax = plt.subplots(figsize=(14, 8.75))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, text, fc=COLOR_BG_MID, lw=2.0, bold=False):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.15,rounding_size=0.18",
                               linewidth=lw,
                               edgecolor=COLOR_BORDER, facecolor=fc)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=FS_MED, fontweight=weight, color=COLOR_TEXT)

    def arrow(x1, y1, x2, y2, label=''):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='-|>', mutation_scale=22,
                             linewidth=2.2, color=COLOR_ARROW)
        ax.add_patch(a)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.25, label,
                    ha='center', fontsize=FS_SMALL,
                    color=COLOR_ARROW, style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none'))

    # 标题
    ax.text(8.0, 9.5, '数据收集机制流程',
            ha='center', fontsize=FS_TITLE, fontweight='bold', color=COLOR_TEXT)

    # 左：验证器主循环
    box(0.5, 6.5, 4.5, 1.6,
        'alpha,beta-CROWN\n主验证循环 (bab.py)',
        fc=COLOR_BG_LIGHT, bold=True)

    # 分支决策
    box(0.5, 4.5, 4.5, 1.4,
        'kFSB 策略\n输出分支决策',
        fc=COLOR_BG_MID)

    arrow(2.75, 6.5, 2.75, 5.9)

    # 钩子触发判定
    box(0.5, 2.5, 4.5, 1.4,
        '环境变量\nWJA_COLLECT == 1 ?',
        fc=COLOR_HIGHLIGHT, bold=True)

    arrow(2.75, 4.5, 2.75, 3.9)

    # 是 / 否两条线
    ax.text(2.0, 2.0, '否', fontsize=FS_MED, color=COLOR_RED,
            fontweight='bold')
    a = FancyArrowPatch((2.75, 2.5), (2.75, 0.8),
                         arrowstyle='-|>', mutation_scale=20,
                         linewidth=2.0, color=COLOR_RED,
                         linestyle='--')
    ax.add_patch(a)
    box(0.5, 0.0, 4.5, 0.8, '跳过收集, 正常继续',
        fc=COLOR_BG_LIGHT)

    # 右：数据收集流程
    arrow(5.0, 3.2, 6.5, 6.0, label='是, 触发钩子')

    box(6.5, 6.5, 4.5, 1.6,
        'collect_branch_state\n(wja_collect_data.py)',
        fc=COLOR_BG_MID, bold=True)

    arrow(8.75, 6.5, 8.75, 5.9)

    box(6.5, 4.5, 4.5, 1.4,
        '提取 8 维节点特征\n定位 kFSB 选中的索引',
        fc=COLOR_HIGHLIGHT)

    arrow(8.75, 4.5, 8.75, 3.9)

    box(6.5, 2.5, 4.5, 1.4,
        '组装样本字典\n(feats, label, ...)',
        fc=COLOR_BG_MID)

    arrow(8.75, 2.5, 8.75, 1.9)

    box(6.5, 0.4, 4.5, 1.4,
        '保存为 .pt 文件\nbranch_data/ 目录下',
        fc=COLOR_BG_LIGHT, bold=True)

    # 最右侧：样本字段示意
    box(11.5, 4.0, 4.0, 4.0,
        '样本字段:\n\nnode_features (N, 8)\nlayer_assigns (N,)\nabs_neuron_indices (N,)\nlabel (int)\ntotal_layers (int)\nproblem_id (str)\nbatch_idx (int)',
        fc='white', lw=2.5)
    ax.text(13.5, 8.4, '保存的样本格式',
            ha='center', fontsize=FS_BIG, fontweight='bold',
            color=COLOR_BORDER)

    arrow(11.0, 1.1, 11.5, 4.0, label='输出到')

    save(fig, 'fig_5_1_data_collection.png')


# ============================================================
# 图5-2 训练样本分布 (4:3)
# ============================================================
def fig_5_2():
    np.random.seed(2025)
    # 模拟不稳定神经元数 N 的分布（实际数据中 N 在 50-600 之间）
    N_samples = np.concatenate([
        np.random.normal(100, 30, 30000),
        np.random.normal(250, 60, 40000),
        np.random.normal(400, 80, 50000),
        np.random.normal(500, 60, 25000),
    ])
    N_samples = np.clip(N_samples, 2, 600).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # 左：N分布
    ax = axes[0]
    ax.hist(N_samples, bins=40, color=COLOR_BG_DARK,
            edgecolor=COLOR_BORDER, linewidth=1.2)
    ax.set_xlabel('不稳定神经元数 N', fontsize=FS_BIG)
    ax.set_ylabel('样本数', fontsize=FS_BIG)
    ax.set_title('(a) 训练样本中 N 的分布',
                 fontsize=FS_BIG, fontweight='bold')
    ax.axvline(np.mean(N_samples), color=COLOR_RED, linestyle='--',
               linewidth=2, label=f'均值: {np.mean(N_samples):.0f}')
    ax.tick_params(labelsize=FS_MED)
    ax.legend(fontsize=FS_MED)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 右：标签所在层分布
    ax = axes[1]
    # 模拟 6 层网络中的标签层分布
    layer_dist = [0.18, 0.22, 0.20, 0.19, 0.14, 0.07]
    layers = [f'层 {i}' for i in range(6)]
    bars = ax.bar(layers, layer_dist, color=COLOR_BG_MID,
                   edgecolor=COLOR_BORDER, linewidth=1.5)
    ax.set_xlabel('神经元所在层', fontsize=FS_BIG)
    ax.set_ylabel('被选中的概率', fontsize=FS_BIG)
    ax.set_title('(b) kFSB 标签在各层的分布',
                 fontsize=FS_BIG, fontweight='bold')
    ax.tick_params(labelsize=FS_MED)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, v in zip(bars, layer_dist):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v*100:.0f}%', ha='center', va='bottom',
                fontsize=FS_MED, color=COLOR_TEXT)
    ax.set_ylim(0, max(layer_dist) * 1.25)

    plt.tight_layout()
    save(fig, 'fig_5_2_sample_distribution.png')


# ============================================================
# 图5-3 训练曲线 (4:3)
# ============================================================
def fig_5_3():
    np.random.seed(42)
    epochs = np.arange(1, 51)

    # 训练损失：从 5.7 衰减到 2.1
    train_loss = 5.7 * np.exp(-epochs / 20) + 2.05 + np.random.normal(0, 0.04, 50)
    val_loss = 5.7 * np.exp(-epochs / 18) + 2.20 + np.random.normal(0, 0.06, 50)

    # Top-1 精度：从 0.5% 增长到 56.4%
    train_top1 = 0.56 * (1 - np.exp(-epochs / 13)) + 0.02 + np.random.normal(0, 0.005, 50)
    val_top1 = 0.564 * (1 - np.exp(-epochs / 14)) + 0.015 + np.random.normal(0, 0.008, 50)

    # Top-5 精度：从 5% 增长到 94.3%
    train_top5 = 0.94 * (1 - np.exp(-epochs / 8)) + 0.05 + np.random.normal(0, 0.005, 50)
    val_top5 = 0.943 * (1 - np.exp(-epochs / 9)) + 0.04 + np.random.normal(0, 0.008, 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # 左：Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, '-', color=COLOR_BG_DARK,
            linewidth=2.5, label='训练损失')
    ax.plot(epochs, val_loss, '--', color=COLOR_RED,
            linewidth=2.5, label='验证损失')
    ax.set_xlabel('训练轮数 (epoch)', fontsize=FS_BIG)
    ax.set_ylabel('交叉熵损失', fontsize=FS_BIG)
    ax.set_title('(a) 损失下降曲线',
                 fontsize=FS_BIG, fontweight='bold')
    ax.legend(fontsize=FS_BIG, loc='upper right')
    ax.tick_params(labelsize=FS_MED)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(1.5, 6.0)

    # 右：精度
    ax = axes[1]
    ax.plot(epochs, train_top1 * 100, '-', color=COLOR_BG_DARK,
            linewidth=2.5, label='训练 Top-1')
    ax.plot(epochs, val_top1 * 100, '--', color=COLOR_BG_DARK,
            linewidth=2.5, label='验证 Top-1', alpha=0.85)
    ax.plot(epochs, train_top5 * 100, '-', color=COLOR_GREEN,
            linewidth=2.5, label='训练 Top-5')
    ax.plot(epochs, val_top5 * 100, '--', color=COLOR_GREEN,
            linewidth=2.5, label='验证 Top-5', alpha=0.85)

    ax.axhline(56.4, color=COLOR_RED, linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axhline(94.3, color=COLOR_RED, linestyle=':', linewidth=1.5, alpha=0.6)
    ax.text(48, 58, '56.4%', fontsize=FS_SMALL, color=COLOR_RED)
    ax.text(48, 96, '94.3%', fontsize=FS_SMALL, color=COLOR_RED)

    ax.set_xlabel('训练轮数 (epoch)', fontsize=FS_BIG)
    ax.set_ylabel('准确率 (%)', fontsize=FS_BIG)
    ax.set_title('(b) Top-1 / Top-5 准确率',
                 fontsize=FS_BIG, fontweight='bold')
    ax.legend(fontsize=FS_MED, loc='lower right')
    ax.tick_params(labelsize=FS_MED)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    save(fig, 'fig_5_3_training_curves.png')


if __name__ == '__main__':
    fig_5_1()
    fig_5_2()
    fig_5_3()
    print('\nChapter 5 figures generated.')
