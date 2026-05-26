import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 全局设置：Arial、不加粗、线条 ≤ 0.5 pt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.linewidth'] = 0.5 / 72  # 0.5 pt
plt.rcParams['xtick.major.width'] = 0.5 / 72
plt.rcParams['ytick.major.width'] = 0.5 / 72
plt.rcParams['patch.linewidth'] = 0.5 / 72

# 读取数据
df = pd.read_csv('annotated_predict_result_non_H5N1_H5N5_H5.csv')

# 按 Year 和 reassort_seg 分组，计算 prob 均值
heatmap_data = df.pivot_table(
    values='prob',
    index='reassort_seg',
    columns='Year',
    aggfunc='mean'
)

# 绘图
fig, ax = plt.subplots(figsize=(12, 8))

# 修改配色：使用与之前代码相同的红白配色 (Reds colormap)
# Reds colormap 默认是白色（低值）到红色（高值）的渐变
cmap = "Reds"  # 直接使用内置的Reds配色

# 画热图，linewidths=0.5 pt
sns.heatmap(
    heatmap_data,
    cmap=cmap,
    vmin=0,
    vmax=1,
    ax=ax,
    cbar=True,
    linewidths=0.5 / 72,  # 格子分隔线 0.5 pt
    linecolor='white',
    annot=False,
    cbar_kws={
        'shrink': 0.8,
        'ticks': np.linspace(0, 1, 6),
        'drawedges': False
    }
)

# 色条边框 0.5 pt
cbar = ax.collections[0].colorbar
cbar.outline.set_linewidth(0.5 / 72)
cbar.ax.tick_params(labelsize=20,length=2,width=0.5/72,direction='out')

# 坐标轴标签与刻度字号（适中不遮挡）
ax.set_xlabel('Year', fontsize=20, labelpad=8)
ax.set_ylabel('Reassort gene', fontsize=20, labelpad=8)
ax.tick_params(axis='both', labelsize=20)

# 保存
plt.tight_layout()
# plt.savefig('heatmap_year_salt_sag.png', dpi=300, bbox_inches='tight',
#             facecolor='white', edgecolor='none')
plt.show()