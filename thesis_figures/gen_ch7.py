"""第七章图表生成脚本 - 实验评估与分析"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
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
COLOR_GNN = '#2874A6'      # 蓝色 - GNN
COLOR_KFSB = '#CB4335'     # 红色 - kFSB
COLOR_ARROW = '#34495E'
COLOR_TEXT = '#17202A'
COLOR_GREEN = '#27AE60'
COLOR_RED = '#C0392B'

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


# 实验数据
EXPS = {
    'mnist_6_100': {
        'samples': 50,
        'gnn': {'verified': 11, 'unknown': 39, 'avg_all': 60.83, 'avg_safe': 0.21, 'max': 96.94},
        'kfsb': {'verified': 14, 'unknown': 36, 'avg_all': 66.50, 'avg_safe': 3.00, 'max': 102.67},
    },
    'mnist_6_200': {
        'samples': 25,
        'gnn': {'verified': 9, 'unknown': 16, 'avg_all': 49.72, 'avg_safe': 0.33, 'max': 122.78},
        'kfsb': {'verified': 11, 'unknown': 14, 'avg_all': 52.53, 'avg_safe': 4.20, 'max': 105.43},
    },
    'mnist_9_100': {
        'samples': 25,
        'gnn': {'verified': 3, 'unknown': 22, 'avg_all': 62.63, 'avg_safe': 0.93, 'max': 78.42},
        'kfsb': {'verified': 4, 'unknown': 21, 'avg_all': 60.94, 'avg_safe': 3.29, 'max': 78.29},
    },
}


# ============================================================
# 图7-1 验证准确率对比 (16:10)
# ============================================================
def fig_7_1():
    fig, ax = plt.subplots(figsize=(14, 8.75))

    benchmarks = list(EXPS.keys())
    bench_labels = ['mnist_6_100\n(50样本)', 'mnist_6_200\n(25样本)', 'mnist_9_100\n(25样本)']
    gnn_acc = [EXPS[b]['gnn']['verified'] / EXPS[b]['samples'] * 100 for b in benchmarks]
    kfsb_acc = [EXPS[b]['kfsb']['verified'] / EXPS[b]['samples'] * 100 for b in benchmarks]

    x = np.arange(len(benchmarks))
    width = 0.35

    bars1 = ax.bar(x - width/2, gnn_acc, width, label='GNN (本文)',
                    color=COLOR_GNN, edgecolor=COLOR_BORDER, linewidth=2.0)
    bars2 = ax.bar(x + width/2, kfsb_acc, width, label='kFSB (基线)',
                    color=COLOR_KFSB, edgecolor=COLOR_BORDER, linewidth=2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels, fontsize=FS_BIG)
    ax.set_ylabel('验证准确率 (%)', fontsize=FS_BIG)
    ax.set_title('GNN 与 kFSB 在三个基准上的验证准确率对比',
                  fontsize=FS_TITLE, fontweight='bold')
    ax.legend(fontsize=FS_BIG, loc='upper right')
    ax.tick_params(axis='y', labelsize=FS_MED)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 60)

    for bars, vals in [(bars1, gnn_acc), (bars2, kfsb_acc)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                    f'{v:.1f}%', ha='center', va='bottom',
                    fontsize=FS_BIG, fontweight='bold', color=COLOR_TEXT)

    # 差值标注
    for i, b in enumerate(benchmarks):
        diff = gnn_acc[i] - kfsb_acc[i]
        ax.annotate(f'差距: {diff:+.1f} pp',
                     xy=(i, max(gnn_acc[i], kfsb_acc[i]) + 6),
                     ha='center', fontsize=FS_SMALL, color=COLOR_RED,
                     style='italic')

    save(fig, 'fig_7_1_accuracy_comparison.png')


# ============================================================
# 图7-2 全体平均时间对比 (16:10)
# ============================================================
def fig_7_2():
    fig, ax = plt.subplots(figsize=(14, 8.75))

    benchmarks = list(EXPS.keys())
    bench_labels = ['mnist_6_100', 'mnist_6_200', 'mnist_9_100']
    gnn_t = [EXPS[b]['gnn']['avg_all'] for b in benchmarks]
    kfsb_t = [EXPS[b]['kfsb']['avg_all'] for b in benchmarks]

    x = np.arange(len(benchmarks))
    width = 0.35

    bars1 = ax.bar(x - width/2, gnn_t, width, label='GNN (本文)',
                    color=COLOR_GNN, edgecolor=COLOR_BORDER, linewidth=2.0)
    bars2 = ax.bar(x + width/2, kfsb_t, width, label='kFSB (基线)',
                    color=COLOR_KFSB, edgecolor=COLOR_BORDER, linewidth=2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels, fontsize=FS_BIG)
    ax.set_ylabel('全体平均验证时间 (秒)', fontsize=FS_BIG)
    ax.set_title('GNN 与 kFSB 在三个基准上的平均验证时间对比',
                  fontsize=FS_TITLE, fontweight='bold')
    ax.legend(fontsize=FS_BIG, loc='upper right')
    ax.tick_params(axis='y', labelsize=FS_MED)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 80)

    for bars, vals in [(bars1, gnn_t), (bars2, kfsb_t)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                    f'{v:.1f}s', ha='center', va='bottom',
                    fontsize=FS_BIG, fontweight='bold', color=COLOR_TEXT)

    # 变化百分比
    for i, b in enumerate(benchmarks):
        change = (gnn_t[i] - kfsb_t[i]) / kfsb_t[i] * 100
        color = COLOR_GREEN if change < 0 else COLOR_RED
        ax.annotate(f'相对变化: {change:+.1f}%',
                     xy=(i, max(gnn_t[i], kfsb_t[i]) + 6),
                     ha='center', fontsize=FS_SMALL, color=color,
                     style='italic', fontweight='bold')

    save(fig, 'fig_7_2_time_comparison.png')


# ============================================================
# 图7-3 安全实例平均时间对比 (4:3)
# ============================================================
def fig_7_3():
    fig, ax = plt.subplots(figsize=(12, 9))

    benchmarks = list(EXPS.keys())
    bench_labels = ['mnist_6_100', 'mnist_6_200', 'mnist_9_100']
    gnn_t = [EXPS[b]['gnn']['avg_safe'] for b in benchmarks]
    kfsb_t = [EXPS[b]['kfsb']['avg_safe'] for b in benchmarks]

    x = np.arange(len(benchmarks))
    width = 0.35

    bars1 = ax.bar(x - width/2, gnn_t, width, label='GNN (本文)',
                    color=COLOR_GNN, edgecolor=COLOR_BORDER, linewidth=2.0)
    bars2 = ax.bar(x + width/2, kfsb_t, width, label='kFSB (基线)',
                    color=COLOR_KFSB, edgecolor=COLOR_BORDER, linewidth=2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_labels, fontsize=FS_BIG)
    ax.set_ylabel('安全实例平均验证时间 (秒)', fontsize=FS_BIG)
    ax.set_title('成功验证样本的平均处理时间对比',
                  fontsize=FS_TITLE, fontweight='bold')
    ax.legend(fontsize=FS_BIG, loc='upper right')
    ax.tick_params(axis='y', labelsize=FS_MED)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 5.5)

    for bars, vals in [(bars1, gnn_t), (bars2, kfsb_t)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                    f'{v:.2f}s', ha='center', va='bottom',
                    fontsize=FS_BIG, fontweight='bold', color=COLOR_TEXT)

    # 减少百分比
    for i, b in enumerate(benchmarks):
        change = (gnn_t[i] - kfsb_t[i]) / kfsb_t[i] * 100
        ax.annotate(f'{change:+.1f}%',
                     xy=(i, max(gnn_t[i], kfsb_t[i]) + 0.5),
                     ha='center', fontsize=FS_BIG, color=COLOR_GREEN,
                     fontweight='bold')

    save(fig, 'fig_7_3_safe_time_comparison.png')


# ============================================================
# 图7-4 综合性能雷达图 (4:3)
# ============================================================
def fig_7_4():
    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw=dict(polar=True))

    # 五个维度: 验证率、全体均时、安全均时、最大时间、整体效率
    # 全部归一化到 [0, 1]，越大越好
    labels = ['验证率', '速度\n(全体)', '速度\n(安全样本)',
              '最大时间\n控制', '综合\n效率']
    n = len(labels)

    # 计算三个基准的归一化指标 (GNN相对于kFSB)
    def normalize_rel(gnn_val, kfsb_val, higher_better=True):
        """归一化到[0, 1]，0.5代表与kFSB持平"""
        if not higher_better:
            ratio = kfsb_val / gnn_val if gnn_val > 0 else 1.0
        else:
            ratio = gnn_val / kfsb_val if kfsb_val > 0 else 1.0
        return min(1.0, max(0.0, 0.5 * ratio))

    # 对三个基准求平均做一个综合雷达
    gnn_scores = []
    kfsb_scores = []
    for b in EXPS:
        d = EXPS[b]
        gnn_scores.append([
            d['gnn']['verified'] / d['samples'],  # 验证率
            1 - d['gnn']['avg_all'] / 100,         # 速度全体（越小越好，转换）
            1 - d['gnn']['avg_safe'] / 5,          # 速度安全（越小越好，转换）
            1 - d['gnn']['max'] / 150,             # 最大时间控制
            (d['gnn']['verified'] / d['samples']) / (d['gnn']['avg_all'] / 60),  # 综合效率
        ])
        kfsb_scores.append([
            d['kfsb']['verified'] / d['samples'],
            1 - d['kfsb']['avg_all'] / 100,
            1 - d['kfsb']['avg_safe'] / 5,
            1 - d['kfsb']['max'] / 150,
            (d['kfsb']['verified'] / d['samples']) / (d['kfsb']['avg_all'] / 60),
        ])

    gnn_avg = np.mean(gnn_scores, axis=0)
    kfsb_avg = np.mean(kfsb_scores, axis=0)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    gnn_avg = gnn_avg.tolist() + [gnn_avg[0]]
    kfsb_avg = kfsb_avg.tolist() + [kfsb_avg[0]]
    angles += [angles[0]]

    ax.plot(angles, gnn_avg, '-', linewidth=2.5, color=COLOR_GNN,
            label='GNN (本文)')
    ax.fill(angles, gnn_avg, alpha=0.25, color=COLOR_GNN)
    ax.plot(angles, kfsb_avg, '--', linewidth=2.5, color=COLOR_KFSB,
            label='kFSB (基线)')
    ax.fill(angles, kfsb_avg, alpha=0.15, color=COLOR_KFSB)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FS_BIG)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=FS_SMALL)
    ax.grid(True, alpha=0.4)
    ax.set_title('GNN vs kFSB 综合性能雷达图\n(三个基准的平均归一化指标, 离中心越远越好)',
                  fontsize=FS_BIG, fontweight='bold', pad=30)
    ax.legend(fontsize=FS_BIG, loc='upper right',
              bbox_to_anchor=(1.25, 1.05))

    save(fig, 'fig_7_4_radar_chart.png')


# ============================================================
# 图7-5 三组实验汇总热力图 (16:10)
# ============================================================
def fig_7_5():
    fig, ax = plt.subplots(figsize=(14, 6))

    benchmarks = ['mnist_6_100', 'mnist_6_200', 'mnist_9_100']
    metrics = ['验证准确率(%)', '全体平均时间(s)', '安全实例时间(s)', '最大时间(s)']

    # 构造数据矩阵（每个单元格"GNN / kfsb"）
    data = np.zeros((len(metrics), len(benchmarks)))
    text = np.empty((len(metrics), len(benchmarks)), dtype=object)

    for j, b in enumerate(benchmarks):
        d = EXPS[b]
        gnn_acc = d['gnn']['verified'] / d['samples'] * 100
        kfsb_acc = d['kfsb']['verified'] / d['samples'] * 100
        data[0, j] = (gnn_acc - kfsb_acc) / kfsb_acc * 100
        text[0, j] = f'GNN: {gnn_acc:.1f}%\nkFSB: {kfsb_acc:.1f}%\n相对: {(gnn_acc-kfsb_acc)/kfsb_acc*100:+.1f}%'

        data[1, j] = (d['gnn']['avg_all'] - d['kfsb']['avg_all']) / d['kfsb']['avg_all'] * 100
        text[1, j] = f"GNN: {d['gnn']['avg_all']:.1f}s\nkFSB: {d['kfsb']['avg_all']:.1f}s\n相对: {data[1,j]:+.1f}%"

        data[2, j] = (d['gnn']['avg_safe'] - d['kfsb']['avg_safe']) / d['kfsb']['avg_safe'] * 100
        text[2, j] = f"GNN: {d['gnn']['avg_safe']:.2f}s\nkFSB: {d['kfsb']['avg_safe']:.2f}s\n相对: {data[2,j]:+.1f}%"

        data[3, j] = (d['gnn']['max'] - d['kfsb']['max']) / d['kfsb']['max'] * 100
        text[3, j] = f"GNN: {d['gnn']['max']:.1f}s\nkFSB: {d['kfsb']['max']:.1f}s\n相对: {data[3,j]:+.1f}%"

    # 红蓝双色：验证率正值为正面（蓝）；时间负值为正面（蓝）
    norm_data = np.copy(data)
    # 验证率行：正值是好的（蓝色）
    # 时间行：负值是好的（蓝色），需要翻转
    for i in range(1, 4):
        norm_data[i] = -norm_data[i]

    im = ax.imshow(norm_data, cmap='RdBu', aspect='auto',
                    vmin=-50, vmax=50)

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(benchmarks, fontsize=FS_BIG)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=FS_BIG)
    ax.set_title('GNN 相对 kFSB 的性能变化汇总 (蓝色为优, 红色为劣)',
                  fontsize=FS_TITLE, fontweight='bold')

    for i in range(len(metrics)):
        for j in range(len(benchmarks)):
            ax.text(j, i, text[i, j], ha='center', va='center',
                     fontsize=FS_SMALL, color=COLOR_TEXT,
                     fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('相对优势 (%)', fontsize=FS_MED)
    cbar.ax.tick_params(labelsize=FS_SMALL)

    plt.tight_layout()
    save(fig, 'fig_7_5_summary_heatmap.png')


if __name__ == '__main__':
    fig_7_1()
    fig_7_2()
    fig_7_3()
    fig_7_4()
    fig_7_5()
    print('\nChapter 7 figures generated.')
