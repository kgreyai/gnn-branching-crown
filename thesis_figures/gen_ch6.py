"""第六章图表生成脚本 - 系统集成"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

OUT = os.path.dirname(os.path.abspath(__file__))

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
# 图6-1 类继承关系与系统集成 (16:10)
# ============================================================
def fig_6_1():
    fig, ax = plt.subplots(figsize=(14, 8.75))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def class_box(x, y, w, h, name, methods, fc=COLOR_BG_MID):
        # 类标题区域
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.1,rounding_size=0.1",
                               linewidth=2.2,
                               edgecolor=COLOR_BORDER, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.3, name, ha='center', va='top',
                fontsize=FS_BIG, fontweight='bold', color=COLOR_TEXT)
        ax.plot([x, x + w], [y + h - 0.6, y + h - 0.6], 'k-',
                linewidth=1.5, color=COLOR_BORDER)
        for i, m in enumerate(methods):
            ax.text(x + 0.2, y + h - 1.0 - i * 0.4, m,
                    fontsize=FS_SMALL, color=COLOR_TEXT)

    def inherit_arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='-|>', mutation_scale=24,
                             linewidth=2.2, color=COLOR_BORDER)
        ax.add_patch(a)

    # 标题
    ax.text(8.0, 9.5, '插件化分支策略接口与类继承',
            ha='center', fontsize=FS_TITLE, fontweight='bold', color=COLOR_TEXT)

    # 基类
    class_box(5.5, 6.5, 5.0, 2.5,
              'NeuronBranchingHeuristic (基类)',
              ['+ get_branching_decisions()',
               '+ find_topk_scores()',
               '+ format_decisions()',
               '# compute_neuron_scores() [虚]'],
              fc=COLOR_BG_LIGHT)

    # 子类：左 BabsrBranching
    class_box(0.3, 2.5, 4.5, 2.5,
              'BabsrBranching',
              ['+ compute_neuron_scores()',
               '  一阶近似评分'],
              fc=COLOR_BG_MID)

    # 子类：中 KfsbBranching
    class_box(5.5, 2.5, 5.0, 2.5,
              'KfsbBranching',
              ['+ compute_neuron_scores()',
               '  top-k 强分支',
               '+ 额外 CROWN 调用'],
              fc=COLOR_BG_MID)

    # 子类：右 GnnBranching (本文新增)
    class_box(11.2, 2.5, 4.5, 2.5,
              'GnnBranching  (本文)',
              ['+ compute_neuron_scores()',
               '+ get_branching_decisions()',
               '+ _try_load_model()',
               '+ _gnn_scores()'],
              fc=COLOR_HIGHLIGHT)

    # 继承箭头
    inherit_arrow(2.5, 5.0, 7.0, 6.5)
    inherit_arrow(8.0, 5.0, 8.0, 6.5)
    inherit_arrow(13.4, 5.0, 9.0, 6.5)

    # 下方说明
    ax.text(8.0, 1.5,
            '所有子类共享 get_branching_decisions 主流程, 仅需实现 compute_neuron_scores',
            ha='center', fontsize=FS_MED, color=COLOR_ARROW,
            style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLOR_BG_LIGHT,
                      edgecolor=COLOR_BORDER, linewidth=1.5))

    ax.text(8.0, 0.5,
            'GnnBranching 额外重写 get_branching_decisions 以支持模型不可用时的自动回退',
            ha='center', fontsize=FS_SMALL, color=COLOR_TEXT)

    save(fig, 'fig_6_1_class_hierarchy.png')


# ============================================================
# 图6-2 在线推理数据流 (16:10)
# ============================================================
def fig_6_2():
    fig, ax = plt.subplots(figsize=(14, 8.75))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, text, fc=COLOR_BG_MID, bold=False):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.15,rounding_size=0.18",
                               linewidth=2.0,
                               edgecolor=COLOR_BORDER, facecolor=fc)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=FS_MED, fontweight=weight, color=COLOR_TEXT)

    def arrow(x1, y1, x2, y2, label='', curve=0):
        if curve == 0:
            a = FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=22,
                                 linewidth=2.2, color=COLOR_ARROW)
        else:
            a = FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=22,
                                 linewidth=2.2, color=COLOR_ARROW,
                                 connectionstyle=f"arc3,rad={curve}")
        ax.add_patch(a)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.25, label,
                    ha='center', fontsize=FS_SMALL, color=COLOR_ARROW,
                    style='italic',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none'))

    # 标题
    ax.text(8.0, 9.5, 'GNN 策略在线推理数据流',
            ha='center', fontsize=FS_TITLE, fontweight='bold', color=COLOR_TEXT)

    # 输入
    box(0.5, 7.0, 3.5, 1.5,
        'BaB 主循环\n传入 domains',
        fc=COLOR_BG_LIGHT, bold=True)

    arrow(4.0, 7.75, 5.0, 7.75)

    # 模型可用判定
    box(5.0, 7.0, 3.5, 1.5,
        'GnnBranching\n模型已加载?',
        fc=COLOR_HIGHLIGHT, bold=True)

    # 模型不可用 -> 回退
    arrow(6.75, 7.0, 6.75, 5.5, label='否')
    box(5.0, 4.5, 3.5, 1.0,
        '回退到 BabsrBranching',
        fc=COLOR_RED + '88')

    # 模型可用 -> 继续
    arrow(8.5, 7.75, 9.5, 7.75, label='是')

    # 特征提取
    box(9.5, 7.0, 3.5, 1.5,
        '逐批次提取节点特征\nfeats (N, 8)',
        fc=COLOR_BG_MID)

    arrow(11.25, 7.0, 11.25, 5.8)

    # GNN 推理
    box(9.5, 4.3, 3.5, 1.5,
        'BranchGNN 前向推理\n输出 scores (N,)',
        fc=COLOR_BG_DARK + 'AA', bold=True)

    arrow(9.5, 5.05, 5.0, 5.05, curve=-0.15)

    # 分数映射
    box(2.5, 4.3, 4.0, 1.5,
        '分数映射\n初始化 all_scores 为 -1e9\n填回对应层张量位置',
        fc=COLOR_HIGHLIGHT)

    arrow(4.5, 4.3, 4.5, 3.0)

    # 基类 top-k 选择
    box(2.5, 1.7, 4.0, 1.3,
        '基类 find_topk_scores()\n选最高分神经元',
        fc=COLOR_BG_MID)

    arrow(6.5, 2.35, 8.5, 2.35)

    # 输出
    box(8.5, 1.7, 4.0, 1.3,
        '输出分支决策\n[layer_idx, neuron_idx]',
        fc=COLOR_BG_LIGHT, bold=True)

    arrow(10.5, 1.7, 10.5, 0.6, label='返回 BaB')
    box(8.5, 0.0, 4.0, 0.5, '继续 beta-CROWN 分支', fc=COLOR_BG_LIGHT)

    # 左侧回退路径
    arrow(6.75, 4.5, 4.5, 3.0, curve=0.15)

    save(fig, 'fig_6_2_online_inference.png')


# ============================================================
# 图6-3 分数初始化的影响 (4:3)
# ============================================================
def fig_6_3():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # 模拟两种初始化方案的结果
    np.random.seed(7)
    n_total = 20
    unstable_idx = [3, 5, 7, 8, 12, 14, 16]  # 不稳定神经元位置
    gnn_scores = np.random.randn(len(unstable_idx)) * 1.5 - 0.3  # 包含负分

    # 方案A：初始化为 0
    scores_A = np.zeros(n_total)
    for i, idx in enumerate(unstable_idx):
        scores_A[idx] = gnn_scores[i]

    # 方案B：初始化为 -1e9（视觉上用很小的负数表示）
    scores_B = np.full(n_total, -10.0)
    for i, idx in enumerate(unstable_idx):
        scores_B[idx] = gnn_scores[i]

    # 左侧：方案A
    ax = axes[0]
    colors_A = [COLOR_GREEN if i in unstable_idx else COLOR_BG_LIGHT
                for i in range(n_total)]
    ax.bar(range(n_total), scores_A, color=colors_A,
           edgecolor=COLOR_BORDER, linewidth=1.5)

    # 标记被选中的（最大值）
    max_idx_A = np.argmax(scores_A)
    ax.axvline(max_idx_A, color=COLOR_RED, linestyle='--',
               linewidth=2.5, alpha=0.7)
    ax.text(max_idx_A, max(scores_A) + 0.3,
            f'选中索引 {max_idx_A}\n(稳定神经元!)',
            ha='center', fontsize=FS_MED, color=COLOR_RED,
            fontweight='bold')

    ax.set_xlabel('神经元索引', fontsize=FS_BIG)
    ax.set_ylabel('得分', fontsize=FS_BIG)
    ax.set_title('(a) 错误方案: 初始化为 0',
                 fontsize=FS_BIG, fontweight='bold')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.tick_params(labelsize=FS_MED)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(min(scores_A) - 1, max(scores_A) + 1.5)

    # 右侧：方案B
    ax = axes[1]
    # 为了可视化，scores_B 限制到 -3 而非真正的 -1e9
    scores_B_vis = np.where(scores_B == -10.0, -3.0, scores_B)
    colors_B = [COLOR_GREEN if i in unstable_idx else COLOR_BG_LIGHT
                for i in range(n_total)]
    bars = ax.bar(range(n_total), scores_B_vis, color=colors_B,
                   edgecolor=COLOR_BORDER, linewidth=1.5)

    # 标注稳定神经元的真实值
    for i in range(n_total):
        if i not in unstable_idx:
            ax.text(i, -2.8, '...', ha='center', va='center',
                    fontsize=FS_SMALL, color=COLOR_RED)

    max_idx_B = unstable_idx[np.argmax(gnn_scores)]
    ax.axvline(max_idx_B, color=COLOR_GREEN, linestyle='--',
               linewidth=2.5, alpha=0.7)
    ax.text(max_idx_B, max(gnn_scores) + 0.3,
            f'选中索引 {max_idx_B}\n(不稳定神经元 OK)',
            ha='center', fontsize=FS_MED, color=COLOR_GREEN,
            fontweight='bold')

    ax.set_xlabel('神经元索引', fontsize=FS_BIG)
    ax.set_ylabel('得分 (稳定神经元位置实际为 -1e9)', fontsize=FS_BIG)
    ax.set_title('(b) 正确方案: 初始化为 -1e9',
                 fontsize=FS_BIG, fontweight='bold')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.tick_params(labelsize=FS_MED)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(-3.3, max(gnn_scores) + 1.5)

    # 全局图例
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLOR_GREEN, edgecolor=COLOR_BORDER,
              label='不稳定神经元 (GNN 实际打分)'),
        Patch(facecolor=COLOR_BG_LIGHT, edgecolor=COLOR_BORDER,
              label='稳定神经元 (无需评分)'),
    ]
    fig.legend(handles=legend_elems, loc='upper center',
               bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=FS_MED)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, 'fig_6_3_score_initialization.png')


if __name__ == '__main__':
    fig_6_1()
    fig_6_2()
    fig_6_3()
    print('\nChapter 6 figures generated.')
