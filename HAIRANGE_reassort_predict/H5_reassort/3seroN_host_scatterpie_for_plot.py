import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import matplotlib.patches as mpatches

# 设置全局样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.3
plt.rcParams['patch.linewidth'] = 0.3
plt.rcParams['axes.edgecolor'] = 'none'  # 隐藏坐标轴边框

# 马卡龙色系定义
# colors = {
#     'Low risk': '#a8d8b9',
#     'Middle risk': '#a8c0d8',
#     'High risk': '#d8a8a8'
# }
colors = {
    'Low risk': '#fcc4ad',
    'Middle risk': '#fa6849',
    'High risk': '#69000d'
}
# 读取数据
df = pd.read_csv('annotated_predict_result_non_H5N1_H5N5_H5.csv')

# 1. 数据预处理
# 提取并排序sero_h（按数字大小排序）
sero_h_unique = sorted(df['Sero_H'].unique(),
                       key=lambda x: int(x.replace('N', '')))
host_unique = sorted(df['Host'].unique())

# 创建坐标映射
x_map = {val: i for i, val in enumerate(sero_h_unique)}
y_map = {val: i for i, val in enumerate(host_unique)}

# 2. 计算每个组合的统计信息
group_stats = []
for sero in sero_h_unique:
    for host in host_unique:
        # 筛选当前组合的数据
        mask = (df['Sero_H'] == sero) & (df['Host'] == host)
        subset = df[mask]

        if len(subset) > 0:
            # 计算risk分布
            risk_counts = subset['Risk'].value_counts()
            total = len(subset)

            # 计算比例
            ratios = {}
            for risk_type in ['Low risk', 'Middle risk', 'High risk']:
                ratios[risk_type] = risk_counts.get(risk_type, 0) / total

            group_stats.append({
                'x': x_map[sero],
                'y': y_map[host],
                'size': total,
                'ratios': ratios,
                'sero': sero,
                'host': host
            })

# 3. 创建图形
fig, ax = plt.subplots(figsize=(14, 10))

# 绘制灰色虚线网格（在饼图下方）
for x in range(len(sero_h_unique)):
    ax.axvline(x=x, color='#cccccc', linestyle='--', linewidth=0.3, zorder=1)
for y in range(len(host_unique)):
    ax.axhline(y=y, color='#cccccc', linestyle='--', linewidth=0.3, zorder=1)

# 气泡大小标准化
max_size = max([stat['size'] for stat in group_stats]) if group_stats else 1
size_scale = 0.35  # 控制气泡大小

# 4. 绘制每个气泡和内部饼图
for stat in group_stats:
    x, y = stat['x'], stat['y']
    bubble_radius = np.sqrt(np.sqrt(stat['size'] / max_size) * size_scale)/1.3

    # 绘制气泡背景（白色填充，灰色边框）
    circle = Circle((x, y), bubble_radius,
                    facecolor='white', edgecolor='#999999',
                    linewidth=0.3, zorder=3)
    ax.add_patch(circle)

    # 绘制内部饼图
    start_angle = 90
    for risk_type, ratio in stat['ratios'].items():
        if ratio > 0:
            wedge = Wedge((x, y), bubble_radius,
                          start_angle, start_angle + ratio * 360,
                          facecolor=colors[risk_type], edgecolor='white',
                          linewidth=0.2, zorder=4)
            ax.add_patch(wedge)
            start_angle += ratio * 360

# 5. 设置坐标轴
ax.set_xlim(-0.5, len(sero_h_unique) - 0.5)
ax.set_ylim(-0.5, len(host_unique) - 0.5)

ax.set_xticks(range(len(sero_h_unique)))
ax.set_xticklabels(sero_h_unique, fontsize=20)
ax.set_yticks(range(len(host_unique)))
ax.set_yticklabels(host_unique, fontsize=20)

# 隐藏刻度线
ax.tick_params(axis='both', length=0)

# 6. 添加图例
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['Low risk'],
                       edgecolor='white', linewidth=0.3),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['Middle risk'],
                       edgecolor='white', linewidth=0.3),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['High risk'],
                       edgecolor='white', linewidth=0.3)
]
ax.legend(legend_elements, ['Low risk', 'Middle risk', 'High risk'],
          loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=20,
          frameon=True, fancybox=True, shadow=False)

# 7. 调整布局并保存
plt.tight_layout()
# plt.savefig('bubble_pie_chart_seroN_host_risk.svg', dpi=600, bbox_inches='tight',
#             facecolor='white', edgecolor='none')
plt.show()

# print("气泡饼图已生成并保存为: bubble_pie_chart.png")
print(f"X轴类别: {sero_h_unique}")
print(f"Y轴类别: {host_unique}")