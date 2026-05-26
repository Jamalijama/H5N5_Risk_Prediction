import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
from scipy.stats import mannwhitneyu  # 导入检验模块

# -------------------------- 1. 配置参数 --------------------------
# 原始H5N5文件（H5N5组）
csv_files_h5n5 = {
    'PB2 (H5N5)': 'test_H5N5pb2_1150+0.03_0.2_resnet34_fc_H5N5_all.csv',
    'PB1 (H5N5)': 'test_H5N5pb1_1150+0.03_0.2_resnet34_trm_fc_H5N5_all.csv',
    'PA (H5N5)': 'test_H5N5pa_1150+0.03_0.2_resnet34_fc_H5N5_all.csv',
    'NP (H5N5)': 'test_H5N5np_1150+0.03_0.2_resnet34_trm_fc_H5N5_all.csv'
}
# 新增H5文件（Other H5组）
csv_files_h5 = {
    'PB2 (Other H5)': '../H5_reassort/test_H5pb2_1150+0.03_0.2_resnet34_fc_H5_all.csv',
    'PB1 (Other H5)': '../H5_reassort/test_H5pb1_1150+0.03_0.2_resnet34_trm_fc_H5_all.csv',
    'PA (Other H5)': '../H5_reassort/test_H5pa_1150+0.03_0.2_resnet34_fc_H5_all.csv',
    'NP (Other H5)': '../H5_reassort/test_H5np_1150+0.03_0.2_resnet34_trm_fc_H5_all.csv'
}

# 严格沿用你原始的配色方案，无任何修改
colors = {
    'PB2': "#4198AC",
    'PB1': "#7BC0CD",
    'PA': "#ED8D5A",
    'NP': "#ECB66C"
}
target_id = "A/Washington/2148/2025"
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 0.5

# -------------------------- 2. 数据加载与预处理 --------------------------
all_data = []
# 处理H5N5组数据
for name in csv_files_h5n5.keys():
    file = csv_files_h5n5[name]
    protein = name.split(' ')[0]  # 提取蛋白类型
    df = pd.read_csv(file)
    epsilon = 1e-10
    df['prob_adjusted'] = df['prob'] + epsilon
    df['prob_log'] = np.log10(df['prob_adjusted'])
    df['prob_normalized'] = df['prob_log']
    df['group'] = name
    df['is_target'] = df['id'] == target_id
    df['type'] = 'H5N5'  # 改为H5N5
    df['protein'] = protein
    all_data.append(df)

# 处理Other H5组数据
for name in csv_files_h5.keys():
    file = csv_files_h5[name]
    protein = name.split(' ')[0]  # 提取蛋白类型
    df = pd.read_csv(file)
    epsilon = 1e-10
    df['prob_adjusted'] = df['prob'] + epsilon
    df['prob_log'] = np.log10(df['prob_adjusted'])
    df['prob_normalized'] = df['prob_log']
    df['group'] = name
    df['is_target'] = False
    df['type'] = 'Other H5'  # 改为Other H5
    df['protein'] = protein
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
target_data = data[data['is_target']].copy()

# 定义x轴顺序：按蛋白成对排列
protein_order = ['PB2', 'PB1', 'PA', 'NP']
group_order = []
for p in protein_order:
    group_order.append(f'{p} (H5N5)')  # 改为H5N5
    group_order.append(f'{p} (Other H5)')  # 改为Other H5
data['group'] = pd.Categorical(data['group'], categories=group_order, ordered=True)


# -------------------------- 3. 定义显著性标注函数 --------------------------
def add_significance(ax, data, protein, x1, x2, y_max, y_offset=0.05):
    """
    在两个分组之间添加显著性标记
    ax: 绘图轴
    data: 数据框
    protein: 蛋白名称
    x1/x2: 两组的x坐标
    y_max: 数据最大值
    y_offset: 标记高度偏移
    """
    # 提取两组数据
    h5n5_data = data[(data['protein'] == protein) & (data['type'] == 'H5N5')]['prob_normalized']  # 改为H5N5
    other_h5_data = data[(data['protein'] == protein) & (data['type'] == 'Other H5')]['prob_normalized']  # 改为Other H5

    # 曼-惠特尼U检验
    stat, p_val = mannwhitneyu(h5n5_data, other_h5_data, alternative='two-sided')

    # 定义显著性标记
    if p_val >= 0.05:
        sig_label = 'ns'
    elif p_val >= 0.01:
        sig_label = '*'
    elif p_val >= 0.001:
        sig_label = '**'
    else:
        sig_label = '***'

    # 绘制横线和标记
    y_pos = y_max + y_offset
    ax.plot([x1, x2], [y_pos, y_pos], lw=1.2, color='#264653')
    ax.text((x1 + x2) / 2, y_pos + 0.02, sig_label, ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#264653')
    return stat, p_val


# -------------------------- 4. 绘图 --------------------------
fig, ax = plt.subplots(figsize=(16, 8))

# 绘制H5N5组小提琴图：直接按蛋白匹配原始颜色，确保无偏差
h5n5_groups = [f'{p} (H5N5)' for p in protein_order]  # 改为H5N5
h5n5_colors = ['#4198AC', '#4198AC', '#7BC0CD', '#7BC0CD', '#ED8D5A', '#ED8D5A', '#ECB66C', '#ECB66C']
sns.violinplot(
    data=data[data['group'].isin(h5n5_groups)],
    x='group',
    y='prob_normalized',
    palette=h5n5_colors,
    inner=None,
    width=0.7,
    alpha=0.8,
    ax=ax,
    cut=0,
    linewidth=0
)

# 绘制Other H5组小提琴图：无填充，描边颜色匹配原始配色
other_h5_groups = [f'{p} (Other H5)' for p in protein_order]  # 改为Other H5
other_h5_colors = [colors[p] for p in protein_order]
sns.violinplot(
    data=data[data['group'].isin(other_h5_groups)],
    x='group',
    y='prob_normalized',
    palette=other_h5_colors,
    inner=None,
    width=0.7,
    alpha=0,
    ax=ax,
    cut=0,
    linewidth=0.5,
    edgecolor=other_h5_colors
)

# 绘制箱线图：统一样式
sns.boxplot(
    data=data,
    x='group',
    y='prob_normalized',
    width=0.15,
    boxprops={'facecolor': 'white', 'alpha': 0.7, 'linewidth': 0.5},
    medianprops={'color': '#264653', 'linewidth': 0.5},
    whiskerprops={'color': '#666666', 'linewidth': 0.5},
    capprops={'color': '#666666', 'linewidth': 0.5},
    flierprops={'marker': 'x', 'markersize': 5, 'markeredgewidth': 0.5},
    ax=ax
)

# 绘制散点：按蛋白匹配原始颜色
# H5N5组散点
for p in protein_order:
    subset = data[(data['protein'] == p) & (data['type'] == 'H5N5')]  # 改为H5N5
    ax.scatter(
        x=subset['group'],
        y=subset['prob_normalized'],
        color=colors[p],
        alpha=0.4,
        s=30,
        edgecolors='none',
        zorder=2
    )
# Other H5组散点
for p in protein_order:
    subset = data[(data['protein'] == p) & (data['type'] == 'Other H5')]  # 改为Other H5
    ax.scatter(
        x=subset['group'],
        y=subset['prob_normalized'],
        color=colors[p],
        alpha=0.2,
        s=30,
        edgecolors='none',
        zorder=2
    )

# 绘制目标样本五角星
for idx, (_, row) in enumerate(target_data.iterrows()):
    ax.scatter(
        x=row['group'],
        y=row['prob_normalized'],
        marker='*',
        s=300,
        color='#FFD700',
        alpha=1,
        zorder=5,
        label='Target' if idx == 0 else ""
    )
    ax.text(
        x=row['group'],
        y=row['prob_normalized'] + 0.03,
        s=target_id,
        fontsize=10,
        fontweight='bold',
        color='#264653',
        ha='center',
        va='bottom',
        zorder=5
    )

# 绘制平均值线
for i, name in enumerate(group_order):
    subset = data[data['group'] == name]
    if len(subset) == 0:
        continue
    mean_val = subset['prob_normalized'].mean()
    ax.axhline(
        y=mean_val,
        xmin=i / 8 + 0.05,
        xmax=(i + 1) / 8 - 0.05,
        linestyle='--',
        color='#999999',
        alpha=0.7,
        linewidth=0.5
    )

# -------------------------- 5. 添加显著性标注 --------------------------
# 计算y轴最大值，确定标注高度
y_max_global = data['prob_normalized'].max()
y_offset = y_max_global * 0.08  # 调整偏移量避免遮挡数据

# 为每个蛋白的两组数据添加标注
test_results = {}
for i, protein in enumerate(protein_order):
    x1 = 2 * i  # H5N5组x坐标
    x2 = 2 * i + 1  # Other H5组x坐标
    stat, p_val = add_significance(ax, data, protein, x1, x2, y_max_global, y_offset)
    test_results[protein] = (stat, p_val)

# 调整y轴范围，确保标注完整显示
ax.set_ylim(bottom=data['prob_normalized'].min() - y_offset,
            top=y_max_global + 2 * y_offset)

# -------------------------- 6. 图表美化 --------------------------
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

ax.set_xlabel("", fontsize=14, fontweight='bold', color='#264653')
ax.set_ylabel("Log-transformed Probability (log10 scale)", fontsize=14, fontweight='bold', color='#264653')
ax.set_title("")

# 自定义图例 - 改为H5N5和Other H5
legend_elements = [
    Patch(facecolor='gray', alpha=0.8, label='H5N5'),
    Patch(facecolor='none', edgecolor='gray', linewidth=0.5, label='Other H5'),
    Patch(facecolor=colors['PB2'], label='PB2'),
    Patch(facecolor=colors['PB1'], label='PB1'),
    Patch(facecolor=colors['PA'], label='PA'),
    Patch(facecolor=colors['NP'], label='NP'),
    plt.Line2D([0], [0], marker='*', color='#FFD700', label='Target', markersize=15, linestyle='None')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)

ax.grid(axis='y', linestyle='--', alpha=0.5, color='#E0E0E0', linewidth=0.5)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

# -------------------------- 7. 保存与显示 --------------------------
plt.savefig("H5N5_vs_Other_H5_predict_violin_plot_with_significance.pdf", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# -------------------------- 8. 输出统计检验结果 --------------------------
print("\n=== 曼-惠特尼U检验结果汇总 ===")
for protein in protein_order:
    stat, p_val = test_results[protein]
    print(f"\n{protein}:")
    print(f"  U统计量: {stat:.2f}")
    print(f"  p值: {p_val:.6f}")
    if p_val >= 0.05:
        print(f"  显著性: 不显著 (ns)")
    elif p_val >= 0.01:
        print(f"  显著性: 显著 (*, p<0.05)")
    elif p_val >= 0.001:
        print(f"  显著性: 极显著 (**, p<0.01)")
    else:
        print(f"  显著性: 高度极显著 (***, p<0.001)")

# -------------------------- 9. 输出处理摘要 --------------------------
print("\n=== H5N5组 处理摘要 ===")
for name in csv_files_h5n5.keys():
    subset = data[(data['group'] == name) & (data['type'] == 'H5N5')]  # 改为H5N5
    print(f"\n{name}:")
    print(f"  原始prob范围: {subset['prob'].min():.6f} - {subset['prob'].max():.6f}")
    print(f"  对数转换后范围: {subset['prob_normalized'].min():.3f} - {subset['prob_normalized'].max():.3f}")
    print(f"  平均值: {subset['prob_normalized'].mean():.3f}")

print("\n=== Other H5组 处理摘要 ===")  # 改为Other H5
for name in csv_files_h5.keys():
    subset = data[(data['group'] == name) & (data['type'] == 'Other H5')]  # 改为Other H5
    print(f"\n{name}:")
    print(f"  原始prob范围: {subset['prob'].min():.6f} - {subset['prob'].max():.6f}")
    print(f"  对数转换后范围: {subset['prob_normalized'].min():.3f} - {subset['prob_normalized'].max():.3f}")
    print(f"  平均值: {subset['prob_normalized'].mean():.3f}")

if not target_data.empty:
    print(f"\n目标样本 {target_id} 信息:")
    for _, row in target_data.iterrows():
        print(f"  分组: {row['group']}")
        print(f"  对数转换后值: {row['prob_normalized']:.4f}")  # 改为对数转换后
else:
    print(f"\n警告: 未找到目标样本 {target_id}")