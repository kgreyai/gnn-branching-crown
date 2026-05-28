"""生成毕设论文所有图表，输出到 thesis_figures/ 目录"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Wedge
from matplotlib.lines import Line2D
import numpy as np

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved: {path}')


# ============================================================
# 图1-1 对抗样本示例
# ============================================================
def fig_1_1():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    np.random.seed(42)
    img = np.zeros((28, 28))
    for i in range(8, 20):
        img[i, 12:16] = 1.0
    for i in range(8, 12):
        img[i, 10:18] = 1.0

    noise = np.random.uniform(-0.05, 0.05, (28, 28))
    adv = np.clip(img + noise, 0, 1)

    axes[0].imshow(img, cmap='gray_r')
    axes[0].set_title('原始图像 x0\n预测: 数字 7', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(noise, cmap='RdBu', vmin=-0.1, vmax=0.1)
    axes[1].set_title('添加微小扰动 d\n||d||_inf <= eps = 0.05', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(adv, cmap='gray_r')
    axes[2].set_title('对抗样本 x0 + d\n预测: 数字 1 (错误)', fontsize=11)
    axes[2].axis('off')

    plt.tight_layout()
    save(fig, 'fig_1_1_adversarial_example.png')


# ============================================================
# 图1-2 国内外研究现状脉络图
# ============================================================
def fig_1_2():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def box(x, y, w, h, text, fc='white'):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.1",
                               linewidth=1.5,
                               edgecolor='black', facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, wrap=True)

    box(0.3, 4.5, 3.5, 1.8,
        '神经网络形式化验证方法\n\n精确求解  ->  线性松弛\n->  分支定界  ->  alpha,beta-CROWN')
    box(4.3, 4.5, 3.5, 1.8,
        '分支定界中的分支策略\n\nBaBSR评分  ->  kFSB\n启发式  ->  学习型策略')
    box(8.3, 4.5, 3.5, 1.8,
        '图神经网络在组合优化\n\nGCN / GAT / GIN\n->  MIP分支学习')

    box(2.5, 0.8, 7, 1.6,
        '本课题: 基于图神经网络的验证器分支策略\n'
        '把分支定界中的状态建模为图, 用GNN学习分支决策')

    arrow = FancyArrowPatch((2.0, 4.5), (5.0, 2.4),
                             arrowstyle='->', mutation_scale=18,
                             linewidth=1.3, color='black')
    ax.add_patch(arrow)
    arrow = FancyArrowPatch((6.0, 4.5), (6.0, 2.4),
                             arrowstyle='->', mutation_scale=18,
                             linewidth=1.3, color='black')
    ax.add_patch(arrow)
    arrow = FancyArrowPatch((10.0, 4.5), (7.0, 2.4),
                             arrowstyle='->', mutation_scale=18,
                             linewidth=1.3, color='black')
    ax.add_patch(arrow)

    arrow = FancyArrowPatch((3.8, 5.4), (4.3, 5.4),
                             arrowstyle='->', mutation_scale=15,
                             linewidth=1.0, color='gray', linestyle='--')
    ax.add_patch(arrow)
    ax.text(4.05, 5.6, '需求', fontsize=8, ha='center', color='gray')

    arrow = FancyArrowPatch((8.3, 5.4), (7.8, 5.4),
                             arrowstyle='->', mutation_scale=15,
                             linewidth=1.0, color='gray', linestyle='--')
    ax.add_patch(arrow)
    ax.text(8.05, 5.6, '方法借鉴', fontsize=8, ha='center', color='gray')

    save(fig, 'fig_1_2_research_landscape.png')


# ============================================================
# 图1-3 本文研究技术路线图
# ============================================================
def fig_1_3():
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 11)
    ax.axis('off')

    stages = [
        '图结构建模与特征工程\n(8维节点特征 + 层级结构)',
        'BranchGNN 模型设计\n(GIN风格 + 双向消息传递)',
        '数据收集与模型训练\n(模仿 kFSB, 14万样本)',
        '系统集成与回退机制\n(插件化接入 alpha,beta-CROWN)',
        '实验评估与对比分析\n(MNIST 系列基准, vs kFSB)',
    ]

    y_top = 10.0
    h = 1.4
    gap = 0.4

    for i, text in enumerate(stages):
        y = y_top - i * (h + gap)
        rect = FancyBboxPatch((1.5, y - h), 5, h,
                               boxstyle="round,pad=0.1",
                               linewidth=1.5,
                               edgecolor='black', facecolor='white')
        ax.add_patch(rect)
        ax.text(4, y - h/2, text, ha='center', va='center', fontsize=11)
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((4, y - h - 0.05),
                                     (4, y - h - gap + 0.05),
                                     arrowstyle='->', mutation_scale=20,
                                     linewidth=1.5, color='black')
            ax.add_patch(arrow)

    save(fig, 'fig_1_3_technical_roadmap.png')


# ============================================================
# 图2-1 ReLU三种状态
# ============================================================
def fig_2_1():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    titles = ['稳定激活 (lb >= 0)', '稳定抑制 (ub <= 0)', '不稳定 (lb < 0 < ub)']
    ranges = [(0.5, 3.0), (-3.0, -0.5), (-2.0, 2.5)]

    for ax, title, (lb, ub) in zip(axes, titles, ranges):
        z = np.linspace(-4, 4, 400)
        relu = np.maximum(0, z)
        ax.plot(z, relu, 'k-', linewidth=2, label='h_out = max(0, z)')

        ax.axvspan(lb, ub, alpha=0.25, color='gray',
                   label=f'[lb, ub] = [{lb}, {ub}]')

        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1, 4)
        ax.set_xlabel('z (ReLU输入)', fontsize=10)
        ax.set_ylabel('h_out (ReLU输出)', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    save(fig, 'fig_2_1_relu_states.png')


# ============================================================
# 图2-2 分支定界算法流程图
# ============================================================
def fig_2_2():
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    def box(x, y, w, h, text, shape='rect'):
        if shape == 'oval':
            rect = FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.1,rounding_size=0.3",
                                   linewidth=1.5,
                                   edgecolor='black', facecolor='white')
        elif shape == 'diamond':
            cx, cy = x + w/2, y + h/2
            poly = plt.Polygon(
                [(cx, y+h), (x+w, cy), (cx, y), (x, cy)],
                fill=True, facecolor='white', edgecolor='black', linewidth=1.5)
            ax.add_patch(poly)
            ax.text(cx, cy, text, ha='center', va='center', fontsize=9)
            return
        else:
            rect = FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.05",
                                   linewidth=1.5,
                                   edgecolor='black', facecolor='white')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)

    def arrow(x1, y1, x2, y2, text=''):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='->', mutation_scale=15,
                             linewidth=1.2, color='black')
        ax.add_patch(a)
        if text:
            ax.text((x1+x2)/2 + 0.2, (y1+y2)/2, text, fontsize=8, color='black')

    box(3, 13, 4, 0.7, '开始: 输入域 C 入队', 'oval')

    box(3, 11.6, 4, 0.7, '从队列取出一个子域 D')
    arrow(5, 13, 5, 12.3)

    box(3, 9.9, 4, 0.7, 'CROWN 计算下界 lb')
    arrow(5, 11.6, 5, 10.6)

    box(3, 8.0, 4, 1.2, '判定: lb > 0 ?', 'diamond')
    arrow(5, 9.9, 5, 9.2)

    box(7.3, 8.3, 2.4, 0.6, '剪枝')
    arrow(7, 8.6, 7.3, 8.6, '是')

    box(3, 6.0, 4, 1.2, '判定: 找到反例 ?', 'diamond')
    arrow(5, 8.0, 5, 7.2, '否')

    box(7.3, 6.3, 2.4, 0.6, '输出 unsafe')
    arrow(7, 6.6, 7.3, 6.6, '是')

    box(3, 4.5, 4, 0.7, '分支策略选择神经元 i')
    arrow(5, 6.0, 5, 5.2, '否')

    box(0.5, 2.8, 3, 0.8, '子域 D+ : z_i >= 0')
    box(6, 2.8, 3, 0.8, '子域 D- : z_i <= 0')
    arrow(5, 4.5, 2.0, 3.6)
    arrow(5, 4.5, 7.5, 3.6)

    box(3, 1.3, 4, 0.7, '两个子域加入队列')
    arrow(2.0, 2.8, 3.5, 2.0)
    arrow(7.5, 2.8, 6.5, 2.0)

    arrow(3, 1.65, 0.5, 1.65)
    arrow(0.5, 1.65, 0.5, 11.95)
    arrow(0.5, 11.95, 3, 11.95)

    box(3, 0.0, 4, 0.7, '队列空 -> 输出 safe', 'oval')

    save(fig, 'fig_2_2_bab_flowchart.png')


# ============================================================
# 图2-3 CROWN对ReLU的线性松弛
# ============================================================
def fig_2_3():
    fig, ax = plt.subplots(figsize=(7, 6))

    z = np.linspace(-4, 4, 400)
    relu = np.maximum(0, z)

    lb, ub = -2.0, 3.0
    alpha = 0.4
    z_mid = np.linspace(lb, ub, 100)
    lower_bound = alpha * z_mid
    upper_bound = (ub * z_mid - ub * lb) / (ub - lb)

    ax.fill_between(z_mid, lower_bound, upper_bound,
                    alpha=0.18, color='gray', label='松弛误差区域')

    ax.plot(z, relu, 'k-', linewidth=2.5, label='h_out = max(0, z) (真实 ReLU)')
    ax.plot(z_mid, lower_bound, 'b--', linewidth=2,
            label='下界: alpha * z')
    ax.plot(z_mid, upper_bound, 'r--', linewidth=2,
            label='上界: (ub*z - ub*lb) / (ub - lb)')

    ax.axvline(lb, color='gray', linestyle=':', linewidth=1)
    ax.axvline(ub, color='gray', linestyle=':', linewidth=1)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    ax.text(lb, -0.4, 'lb', ha='center', fontsize=11)
    ax.text(ub, -0.4, 'ub', ha='center', fontsize=11)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-1, 4)
    ax.set_xlabel('z (ReLU输入)', fontsize=11)
    ax.set_ylabel('h_out (ReLU输出)', fontsize=11)
    ax.set_title('CROWN 对不稳定 ReLU 的线性松弛', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    save(fig, 'fig_2_3_crown_relaxation.png')


# ============================================================
# 图2-4 消息传递机制
# ============================================================
def fig_2_4():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (ax, title, k) in enumerate(zip(
        axes,
        ['第 k-1 轮: 节点 v 还未聚合邻居', '第 k 轮: 节点 v 完成聚合更新'],
        [0, 1]
    )):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axis('off')
        ax.set_title(title, fontsize=11)

        center = (0, 0)
        neighbors = [(-2, 1.5), (-2, -1.5), (2, 1.5), (2, -1.5)]

        if k == 1:
            for nx, ny in neighbors:
                arrow = FancyArrowPatch((nx*0.7, ny*0.7), (0.3, 0.0),
                                         arrowstyle='->', mutation_scale=18,
                                         linewidth=1.2, color='red', alpha=0.7)
                ax.add_patch(arrow)

        for i, (nx, ny) in enumerate(neighbors):
            ax.plot([center[0], nx], [center[1], ny], 'k-',
                    linewidth=1.0, alpha=0.5, zorder=1)
            c = Circle((nx, ny), 0.35, facecolor='lightblue',
                       edgecolor='black', linewidth=1.5, zorder=2)
            ax.add_patch(c)
            ax.text(nx, ny, f'u{i+1}', ha='center', va='center',
                    fontsize=10, zorder=3)

        center_color = 'lightgreen' if k == 1 else 'white'
        c = Circle(center, 0.45, facecolor=center_color,
                   edgecolor='black', linewidth=2, zorder=2)
        ax.add_patch(c)
        ax.text(0, 0, 'v', ha='center', va='center', fontsize=12,
                fontweight='bold', zorder=3)

        if k == 1:
            ax.text(0, -2.4,
                    "h_v_new = UPDATE( h_v_old, AGG({ h_u : u in N(v) }) )",
                    ha='center', fontsize=9, family='monospace')

    plt.tight_layout()
    save(fig, 'fig_2_4_message_passing.png')


# ============================================================
# 图2-5 GCN / GAT / GIN 对比
# ============================================================
def fig_2_5():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    cases = [
        ('GCN: 对称归一化平均',
         ['1/sqrt(d*d1)', '1/sqrt(d*d2)', '1/sqrt(d*d3)'],
         '表达能力: 较弱\n参数: W\n开销: 低'),
        ('GAT: 注意力加权平均',
         ['a1 = 0.5', 'a2 = 0.2', 'a3 = 0.3'],
         '表达能力: 中等\n参数: W, a\n开销: 高'),
        ('GIN: 求和 + MLP',
         ['1', '1', '1'],
         '表达能力: 最强\n参数: W1, W2, eps\n开销: 中'),
    ]

    for ax, (title, edge_labels, info) in zip(axes, cases):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')

        center = (0, 0)
        neighbors = [(-1.8, 1.5), (1.8, 1.5), (0, -2)]
        for i, ((nx, ny), label) in enumerate(zip(neighbors, edge_labels)):
            ax.plot([center[0], nx], [center[1], ny], 'k-',
                    linewidth=1.2, alpha=0.6, zorder=1)
            mid_x, mid_y = (center[0] + nx) / 2, (center[1] + ny) / 2
            ax.text(mid_x + 0.15, mid_y, label, fontsize=8,
                    color='darkred', zorder=4)
            c = Circle((nx, ny), 0.32, facecolor='lightblue',
                       edgecolor='black', linewidth=1.3, zorder=2)
            ax.add_patch(c)
            ax.text(nx, ny, f'u{i+1}', ha='center', va='center', fontsize=10)

        c = Circle(center, 0.4, facecolor='lightgreen',
                   edgecolor='black', linewidth=2, zorder=2)
        ax.add_patch(c)
        ax.text(0, 0, 'v', ha='center', va='center', fontsize=12,
                fontweight='bold')

        ax.text(0, 2.6, info, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save(fig, 'fig_2_5_gnn_comparison.png')


if __name__ == '__main__':
    fig_1_1()
    fig_1_2()
    fig_1_3()
    fig_2_1()
    fig_2_2()
    fig_2_3()
    fig_2_4()
    fig_2_5()
    print('\nAll figures generated successfully.')
