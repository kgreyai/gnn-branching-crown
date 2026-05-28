"""第三章图表生成脚本"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

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
# 图3-1 验证状态到图结构的映射
# ============================================================
def fig_3_1():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左：原始网络结构（多层神经元）
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('(a) 原始神经网络结构', fontsize=12)

    n_layers = 4
    n_per_layer = [4, 5, 5, 4]
    layer_x = np.linspace(1, 9, n_layers)
    # 标记哪些是不稳定神经元
    unstable_flag = {
        0: [],
        1: [1, 3],
        2: [0, 2, 4],
        3: [2],
    }

    for l in range(n_layers):
        y_positions = np.linspace(1.5, 6.5, n_per_layer[l])
        for n in range(n_per_layer[l]):
            color = '#ffaaaa' if n in unstable_flag[l] else 'lightgray'
            edge = 'red' if n in unstable_flag[l] else 'gray'
            lw = 2 if n in unstable_flag[l] else 1
            c = Circle((layer_x[l], y_positions[n]), 0.25,
                       facecolor=color, edgecolor=edge, linewidth=lw, zorder=2)
            ax.add_patch(c)

            if l < n_layers - 1:
                next_y_positions = np.linspace(1.5, 6.5, n_per_layer[l+1])
                for n2 in range(n_per_layer[l+1]):
                    ax.plot([layer_x[l], layer_x[l+1]],
                            [y_positions[n], next_y_positions[n2]],
                            'k-', linewidth=0.3, alpha=0.3, zorder=1)
        ax.text(layer_x[l], 7.3, f'层 {l}', ha='center', fontsize=10)

    legend1 = mpatches.Patch(facecolor='#ffaaaa', edgecolor='red', label='不稳定神经元')
    legend2 = mpatches.Patch(facecolor='lightgray', edgecolor='gray', label='稳定神经元')
    ax.legend(handles=[legend1, legend2], loc='lower center', fontsize=10)

    # 右：抽取后的图结构
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('(b) 抽取后的不稳定神经元图', fontsize=12)

    unstable_nodes = []
    for l in range(n_layers):
        y_positions = np.linspace(1.5, 6.5, n_per_layer[l])
        for n in unstable_flag[l]:
            unstable_nodes.append((layer_x[l], y_positions[n], l, n))

    # 同层节点之间用虚线表示共享层嵌入
    for l in range(n_layers):
        nodes_in_layer = [(x, y) for x, y, lv, _ in unstable_nodes if lv == l]
        if len(nodes_in_layer) >= 2:
            for i in range(len(nodes_in_layer)):
                for j in range(i+1, len(nodes_in_layer)):
                    ax.plot([nodes_in_layer[i][0], nodes_in_layer[j][0]],
                            [nodes_in_layer[i][1], nodes_in_layer[j][1]],
                            'b:', linewidth=0.8, alpha=0.6, zorder=1)

    # 跨层连线（前向/反向消息）
    for l in range(n_layers - 1):
        cur = [(x, y) for x, y, lv, _ in unstable_nodes if lv == l]
        nxt = [(x, y) for x, y, lv, _ in unstable_nodes if lv == l+1]
        for x1, y1 in cur:
            for x2, y2 in nxt:
                ax.annotate('', xy=(x2-0.25, y2), xytext=(x1+0.25, y1),
                            arrowprops=dict(arrowstyle='->', color='green',
                                            lw=0.8, alpha=0.5))

    for x, y, l, n in unstable_nodes:
        c = Circle((x, y), 0.3, facecolor='#ffaaaa', edgecolor='red',
                   linewidth=2, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, f'L{l}\nN{n}', ha='center', va='center', fontsize=7)
        ax.text(x, 7.3, f'层 {l}', ha='center', fontsize=10)

    # 图例
    legend1 = Line2D = mpatches.Patch(color='blue', label='同层共享层嵌入(虚线)')
    legend2 = mpatches.Patch(color='green', label='跨层消息传递(箭头)')
    ax.legend(handles=[legend1, legend2], loc='lower center', fontsize=9)

    plt.tight_layout()
    save(fig, 'fig_3_1_graph_modeling.png')


# ============================================================
# 图3-2 8维节点特征语义分组
# ============================================================
def fig_3_2():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('8维节点特征向量及其语义分组', fontsize=12)

    # 中心节点
    c = Circle((7, 4), 0.6, facecolor='lightgreen',
               edgecolor='black', linewidth=2, zorder=3)
    ax.add_patch(c)
    ax.text(7, 4, '不稳定\n神经元', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # 四组特征
    groups = [
        ('边界特征', [
            ('f0: lb (下界)', '原始下界, 截断到[-10, 0]'),
            ('f1: ub (上界)', '原始上界, 截断到[0, 10]'),
            ('f2: center (中心)', '(lb+ub)/2, 综合位置'),
            ('f3: width (宽度)', 'ub-lb, 不稳定度'),
        ], (1.5, 6.5), 'lightblue'),
        ('松弛几何特征', [
            ('f4: neg_ratio', '-lb/(ub-lb), ReLU松弛斜率'),
        ], (12.5, 6.5), 'lightyellow'),
        ('贡献度特征', [
            ('f5: lA_magnitude', '对输出边界的影响度'),
            ('f7: intercept_score', 'width/4 * lA, 综合启发式'),
        ], (12.5, 1.5), 'lightcoral'),
        ('结构特征', [
            ('f6: layer_idx_norm', 'l/(L-1), 归一化层位置'),
        ], (1.5, 1.5), 'lightpink'),
    ]

    for title, feats, (cx, cy), color in groups:
        # 组标题
        ax.text(cx, cy + 0.8, title, ha='center', fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9,
                          edgecolor='black'))

        # 特征列表
        text_lines = '\n'.join([f'{n}: {d}' for n, d in feats])
        ax.text(cx, cy - 0.5, text_lines, ha='center', va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                          edgecolor=color, linewidth=1.5))

        # 从中心连线到组
        ax.plot([7, cx], [4, cy], 'k-', linewidth=1.2, alpha=0.6, zorder=1)

    save(fig, 'fig_3_2_feature_grouping.png')


# ============================================================
# 图3-3 不稳定神经元判定与mask一致性
# ============================================================
def fig_3_3():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：(lb<0)&(ub>0) 判定
    ax = axes[0]
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title('(a) 基于边界值的判定准则', fontsize=12)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel('lb (神经元下界)', fontsize=10)
    ax.set_ylabel('ub (神经元上界)', fontsize=10)

    # 不稳定区域：lb<0 且 ub>0
    ax.fill_betweenx([0, 4], -4, 0, alpha=0.3, color='red',
                      label='不稳定: lb<0 & ub>0')
    # 稳定激活：lb>=0
    ax.fill_betweenx([0, 4], 0, 4, alpha=0.2, color='green',
                      label='稳定激活: lb>=0')
    # 稳定抑制：ub<=0
    ax.fill_betweenx([-4, 0], -4, 4, alpha=0.2, color='gray',
                      label='稳定抑制: ub<=0')

    # 样例点
    samples = [(-1.5, 2.0, '不稳定'), (1.0, 2.5, '激活'),
               (-2.0, -0.5, '抑制'), (-2.8, 1.5, '不稳定')]
    for x, y, label in samples:
        ax.scatter([x], [y], s=80, c='black', zorder=3)
        ax.annotate(label, (x, y), xytext=(x+0.2, y+0.2), fontsize=9)

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右：mask 与 (lb<0)&(ub>0) 的时序差异
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('(b) mask 与边界判定的时序不一致', fontsize=12)

    # 时间线
    ax.annotate('', xy=(9, 7), xytext=(1, 7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(5, 7.4, '时间轴', ha='center', fontsize=10)

    events = [
        (1.5, 6, 'kFSB 评估候选'),
        (3.5, 5, '选中神经元 i'),
        (5.5, 4, 'mask[i] = 0\n(立即清零)'),
        (7.5, 3, '数据收集触发'),
        (9, 2, 'mask 找不到 i\n但 lb/ub 仍可定位'),
    ]

    for x, y, text in events:
        ax.annotate('', xy=(x, 6.8), xytext=(x, 7),
                    arrowprops=dict(arrowstyle='-', lw=1, color='gray'))
        ax.text(x, y, text, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          edgecolor='black'))

    ax.text(5, 0.5, '结论: 应统一使用 (lb<0) & (ub>0) 作为判定准则',
            ha='center', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='red', linewidth=1.5))

    plt.tight_layout()
    save(fig, 'fig_3_3_unstable_criterion.png')


# ============================================================
# 图3-4 特征提取流程图
# ============================================================
def fig_3_4():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, text, fc='white'):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.1",
                               linewidth=1.5,
                               edgecolor='black', facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='->', mutation_scale=18,
                             linewidth=1.3, color='black')
        ax.add_patch(a)

    # 输入
    box(0.5, 8.5, 11, 0.9,
        '输入: domains 字典  (lower_bounds, upper_bounds, lAs, split_nodes)',
        fc='lightblue')

    # 取批次 b
    box(4, 7, 4, 0.8, '取批次索引 b 对应的数据', fc='white')
    arrow(6, 8.5, 6, 7.8)

    # 遍历层
    box(4, 5.5, 4, 0.8, '按 split_nodes 顺序遍历每一层', fc='white')
    arrow(6, 7, 6, 6.3)

    # 三个并行分支
    box(0.2, 3.5, 3.5, 1.3,
        '提取不稳定神经元\nunstable = (lb<0) & (ub>0)\nabs_indices = unstable.nonzero()',
        fc='#fff5e6')
    box(4.25, 3.5, 3.5, 1.3,
        '构造8维节点特征\nlb, ub, center, width,\nneg_ratio, lA, layer_norm, intercept',
        fc='#fff5e6')
    box(8.3, 3.5, 3.5, 1.3,
        '记录所属层索引\nlayer_assigns = [l, l, l, ...]\n(长度为该层 N_l)',
        fc='#fff5e6')

    arrow(5, 5.5, 2, 4.8)
    arrow(6, 5.5, 6, 4.8)
    arrow(7, 5.5, 10, 4.8)

    # 合并
    box(3, 1.8, 6, 0.9, '所有层结果在 dim=0 上拼接合并', fc='lightyellow')
    arrow(2, 3.5, 5, 2.7)
    arrow(6, 3.5, 6, 2.7)
    arrow(10, 3.5, 7, 2.7)

    # 输出
    box(0.5, 0.2, 11, 0.9,
        '输出: feats(N,8), layer_assigns(N,), abs_neuron_indices(N,), layer_names',
        fc='lightgreen')
    arrow(6, 1.8, 6, 1.1)

    save(fig, 'fig_3_4_feature_extraction.png')


if __name__ == '__main__':
    fig_3_1()
    fig_3_2()
    fig_3_3()
    fig_3_4()
    print('\nChapter 3 figures generated.')
