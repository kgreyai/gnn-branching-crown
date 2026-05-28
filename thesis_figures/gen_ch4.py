"""第四章图表生成脚本 - 高质量学术示意图
要求: 16:10 比例, 16pt+ 字号, 300 DPI, 蓝灰配色
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

OUT = os.path.dirname(os.path.abspath(__file__))

# 蓝灰配色
COLOR_BG_LIGHT = '#EAF2F8'       # 浅蓝灰背景
COLOR_BG_MID = '#AED6F1'         # 中蓝
COLOR_BG_DARK = '#5DADE2'        # 深蓝
COLOR_BORDER = '#1B4F72'         # 深蓝边框
COLOR_HIGHLIGHT = '#F8C471'      # 强调色（沙色）
COLOR_ARROW = '#34495E'          # 深灰箭头
COLOR_TEXT = '#17202A'           # 接近黑色

# 字体大小（满足 >=16pt 要求）
FS_TITLE = 18
FS_BIG = 17
FS_MED = 16
FS_SMALL = 14   # 仅用于注释


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white',
                pad_inches=0.3)
    plt.close(fig)
    print(f'saved: {path}')


# ============================================================
# 图4-1 BranchGNN 模型整体架构 (16:10)
# ============================================================
def fig_4_1():
    fig, ax = plt.subplots(figsize=(14, 8.75))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def stage_box(x, y, w, h, title, detail, fc=COLOR_BG_MID):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.15,rounding_size=0.18",
                               linewidth=2.2,
                               edgecolor=COLOR_BORDER, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.68, title,
                ha='center', va='center',
                fontsize=FS_BIG, fontweight='bold', color=COLOR_TEXT)
        ax.text(x + w/2, y + h*0.28, detail,
                ha='center', va='center',
                fontsize=FS_MED, color=COLOR_TEXT)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='-|>', mutation_scale=22,
                             linewidth=2.2, color=COLOR_ARROW)
        ax.add_patch(a)

    # 输入数据
    stage_box(0.4, 4.2, 2.6, 2.0,
              '输入',
              'feats (N, 8)\nlayer_assigns (N,)',
              fc=COLOR_BG_LIGHT)

    arrow(3.0, 5.2, 3.7, 5.2)

    # 节点编码器
    stage_box(3.7, 4.2, 2.8, 2.0,
              '节点编码器',
              'MLP(8→64)\nN×8 → N×64',
              fc=COLOR_BG_MID)

    arrow(6.5, 5.2, 7.2, 5.2)

    # 消息传递（2轮）— 用大方框包起来
    big = FancyBboxPatch((7.2, 3.4), 5.4, 3.6,
                          boxstyle="round,pad=0.18,rounding_size=0.2",
                          linewidth=2.5, linestyle='--',
                          edgecolor=COLOR_BORDER, facecolor='white')
    ax.add_patch(big)
    ax.text(9.9, 6.6, '双向消息传递 ( × 2 轮 )',
            ha='center', fontsize=FS_BIG, fontweight='bold', color=COLOR_BORDER)

    stage_box(7.5, 4.6, 2.3, 1.4,
              '层聚合',
              'N×64 → L×64',
              fc=COLOR_HIGHLIGHT)
    stage_box(10.1, 4.6, 2.3, 1.4,
              '消息+更新',
              '层间双向消息',
              fc=COLOR_HIGHLIGHT)

    arrow(9.8, 5.3, 10.1, 5.3)
    # 反向回流箭头表示多轮
    a = FancyArrowPatch((11.3, 4.55), (8.6, 4.55),
                         arrowstyle='-|>', mutation_scale=18,
                         linewidth=1.5, color=COLOR_ARROW,
                         connectionstyle="arc3,rad=-0.35", linestyle='--')
    ax.add_patch(a)
    ax.text(9.9, 3.9, '迭代 2 轮', ha='center', fontsize=FS_SMALL,
            color=COLOR_ARROW, style='italic')

    arrow(12.6, 5.2, 13.3, 5.2)

    # 输出头
    stage_box(13.3, 4.2, 2.4, 2.0,
              '输出头',
              'Linear(64→1)\nN×64 → (N,)',
              fc=COLOR_BG_MID)

    # 顶部说明
    ax.text(8.0, 9.0, 'BranchGNN 整体架构',
            ha='center', fontsize=FS_TITLE, fontweight='bold', color=COLOR_TEXT)

    # 底部说明：数据流标注
    ax.text(8.0, 2.0,
            '数据维度变化:  (N, 8)  →  (N, 64)  →  (N, 64)  →  (N,)',
            ha='center', fontsize=FS_MED, color=COLOR_TEXT,
            style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_BG_LIGHT,
                      edgecolor=COLOR_BORDER, linewidth=1.5))

    ax.text(8.0, 0.8,
            '其中 N 为不稳定神经元总数, L 为不稳定神经元跨越的层数',
            ha='center', fontsize=FS_SMALL, color=COLOR_ARROW)

    save(fig, 'fig_4_1_branchgnn_architecture.png')


# ============================================================
# 图4-2 单轮消息传递数据流 (16:10)
# ============================================================
def fig_4_2():
    fig, ax = plt.subplots(figsize=(14, 8.75))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, text, fc=COLOR_BG_MID, bold=False):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.12,rounding_size=0.15",
                               linewidth=2.0,
                               edgecolor=COLOR_BORDER, facecolor=fc)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=FS_MED, fontweight=weight, color=COLOR_TEXT)

    def arrow(x1, y1, x2, y2, label='', curve=0):
        if curve == 0:
            a = FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=20,
                                 linewidth=2.0, color=COLOR_ARROW)
        else:
            a = FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=20,
                                 linewidth=2.0, color=COLOR_ARROW,
                                 connectionstyle=f"arc3,rad={curve}")
        ax.add_patch(a)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.25, label,
                    ha='center', fontsize=FS_SMALL, color=COLOR_ARROW,
                    style='italic')

    # 顶部标题
    ax.text(8.0, 9.4, '单轮消息传递流程',
            ha='center', fontsize=FS_TITLE, fontweight='bold', color=COLOR_TEXT)

    # 输入：节点嵌入 h (N, 64)
    box(0.5, 7.0, 3.0, 1.4,
        '节点嵌入 h\n(N, 64)', fc=COLOR_BG_LIGHT, bold=True)

    # 步骤1：层聚合
    box(4.5, 7.0, 3.0, 1.4,
        '步骤1: 层聚合\n按层均值\n→ (L, 64)', fc=COLOR_HIGHLIGHT)
    arrow(3.5, 7.7, 4.5, 7.7, label='聚合')

    # 步骤2a：前向消息 MLP
    box(8.5, 8.2, 3.0, 1.2,
        '前向消息 MLP\n(L, 64) → (L, 64)', fc=COLOR_BG_MID)
    arrow(7.5, 7.7, 8.5, 8.8, curve=0.15)

    # 步骤2b：反向消息 MLP
    box(8.5, 6.0, 3.0, 1.2,
        '反向消息 MLP\n(L, 64) → (L, 64)', fc=COLOR_BG_MID)
    arrow(7.5, 7.7, 8.5, 6.6, curve=-0.15)

    # 步骤3：层级消息展开到每个节点
    box(12.5, 8.2, 3.0, 1.2,
        '前向消息广播\n→ (N, 64)', fc=COLOR_BG_LIGHT)
    arrow(11.5, 8.8, 12.5, 8.8)

    box(12.5, 6.0, 3.0, 1.2,
        '反向消息广播\n→ (N, 64)', fc=COLOR_BG_LIGHT)
    arrow(11.5, 6.6, 12.5, 6.6)

    # 步骤4：节点更新
    box(6.0, 3.0, 4.0, 1.6,
        '步骤4: 节点更新\nMLP([h, fwd, bkd])\n(N, 192) → (N, 64)',
        fc=COLOR_HIGHLIGHT, bold=True)

    # 三条线汇聚到节点更新
    arrow(2.0, 7.0, 6.5, 4.6, curve=-0.2)
    arrow(14.0, 8.2, 9.5, 4.6, curve=0.25)
    arrow(14.0, 6.0, 9.5, 4.0, curve=-0.25)

    # 标注三股输入
    ax.text(3.2, 5.7, '自身嵌入', fontsize=FS_SMALL,
            color=COLOR_ARROW, style='italic')
    ax.text(12.0, 6.7, '前向消息', fontsize=FS_SMALL,
            color=COLOR_ARROW, style='italic')
    ax.text(12.0, 5.0, '反向消息', fontsize=FS_SMALL,
            color=COLOR_ARROW, style='italic')

    # 输出：更新后的节点嵌入
    box(6.0, 1.0, 4.0, 1.2,
        '更新后嵌入 h_new (N, 64)', fc=COLOR_BG_LIGHT, bold=True)
    arrow(8.0, 3.0, 8.0, 2.2)

    save(fig, 'fig_4_2_message_passing_flow.png')


# ============================================================
# 图4-3 张量形状变化总览 (4:3)
# ============================================================
def fig_4_3():
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')

    def shape_box(x, y, w, h, stage, shape_text, desc, fc=COLOR_BG_MID):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.12,rounding_size=0.18",
                               linewidth=2.2,
                               edgecolor=COLOR_BORDER, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.72, stage, ha='center', va='center',
                fontsize=FS_BIG, fontweight='bold', color=COLOR_TEXT)
        ax.text(x + w/2, y + h*0.45, shape_text, ha='center', va='center',
                fontsize=FS_MED, color=COLOR_BORDER, family='monospace',
                fontweight='bold')
        ax.text(x + w/2, y + h*0.18, desc, ha='center', va='center',
                fontsize=FS_SMALL, color=COLOR_ARROW)

    def down_arrow(x, y1, y2, label=''):
        a = FancyArrowPatch((x, y1), (x, y2),
                             arrowstyle='-|>', mutation_scale=24,
                             linewidth=2.5, color=COLOR_ARROW)
        ax.add_patch(a)
        if label:
            ax.text(x + 0.4, (y1+y2)/2, label,
                    fontsize=FS_MED, color=COLOR_ARROW, va='center')

    # 标题
    ax.text(6.0, 8.5, '张量形状变化总览',
            ha='center', fontsize=FS_TITLE, fontweight='bold', color=COLOR_TEXT)

    # 5 个阶段，从上往下
    stages = [
        (7.4, '输入', '(N, 8)', '原始 8 维节点特征', COLOR_BG_LIGHT),
        (5.7, '编码后', '(N, 64)', '通过节点编码器', COLOR_BG_MID),
        (4.0, '层聚合', '(L, 64)', 'scatter mean 按层', COLOR_HIGHLIGHT),
        (2.3, '更新后', '(N, 64)', '经 2 轮消息传递', COLOR_BG_MID),
        (0.4, '输出得分', '(N,)', '每个不稳定神经元一个标量', COLOR_BG_DARK),
    ]

    arrows_labels = [
        'node_encoder',
        'scatter_mean',
        'fwd_msg + bkd_msg + update_mlp',
        'score_head',
    ]

    for i, (y, name, shape, desc, fc) in enumerate(stages):
        shape_box(3.5, y, 5.0, 1.4, name, shape, desc, fc=fc)
        if i < len(stages) - 1:
            next_y = stages[i+1][0]
            down_arrow(6.0, y, next_y + 1.4, label=arrows_labels[i])

    save(fig, 'fig_4_3_tensor_shape_flow.png')


# ============================================================
# 图4-4 模型参数与复杂度概览 (4:3)
# ============================================================
def fig_4_4():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # 左侧：参数构成柱状图
    ax = axes[0]
    components = ['节点\n编码器', '前向消息\nMLP', '反向消息\nMLP', '更新\nMLP', '输出头']
    params = [4736, 16640, 16640, 16512, 65]
    colors = [COLOR_BG_DARK, COLOR_BG_MID, COLOR_BG_MID,
              COLOR_HIGHLIGHT, COLOR_BG_LIGHT]

    bars = ax.bar(components, params, color=colors,
                   edgecolor=COLOR_BORDER, linewidth=2.0)
    ax.set_ylabel('参数数量', fontsize=FS_BIG)
    ax.set_title('(a) 各模块参数构成', fontsize=FS_BIG, fontweight='bold')
    ax.tick_params(axis='x', labelsize=FS_MED)
    ax.tick_params(axis='y', labelsize=FS_MED)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    total = sum(params)
    for bar, val in zip(bars, params):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, val + 500,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=FS_SMALL,
                fontweight='bold', color=COLOR_TEXT)
    ax.set_ylim(0, max(params) * 1.25)

    # 右侧：与对比方案的复杂度
    ax = axes[1]
    methods = ['BaBSR', 'kFSB\n(k=3)', 'GNN\n(本文)']
    crown_calls = [0, 6, 0]
    gnn_calls = [0, 0, 1]
    width = 0.35
    x = np.arange(len(methods))

    bars1 = ax.bar(x - width/2, crown_calls, width,
                    label='额外 CROWN 边界传播次数',
                    color=COLOR_BG_DARK, edgecolor=COLOR_BORDER, linewidth=2.0)
    bars2 = ax.bar(x + width/2, gnn_calls, width,
                    label='GNN 前向推理次数',
                    color=COLOR_HIGHLIGHT, edgecolor=COLOR_BORDER, linewidth=2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=FS_MED)
    ax.set_ylabel('每次分支决策的额外开销', fontsize=FS_BIG)
    ax.set_title('(b) 不同策略的计算开销对比', fontsize=FS_BIG, fontweight='bold')
    ax.tick_params(axis='y', labelsize=FS_MED)
    ax.legend(fontsize=FS_MED, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 8)

    for bars, vals in [(bars1, crown_calls), (bars2, gnn_calls)]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.15,
                        str(val), ha='center', va='bottom',
                        fontsize=FS_MED, fontweight='bold', color=COLOR_TEXT)

    plt.tight_layout()
    save(fig, 'fig_4_4_complexity_analysis.png')


if __name__ == '__main__':
    fig_4_1()
    fig_4_2()
    fig_4_3()
    fig_4_4()
    print('\nChapter 4 figures generated.')
