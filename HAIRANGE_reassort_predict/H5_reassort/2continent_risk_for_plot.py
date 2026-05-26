import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 全局字体设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'

# 读取数据
df = pd.read_csv("annotated_predict_result_non_H5N1_H5N5_H5.csv")
df_filtered = df[df['Area'] != 'Unknown']  # 按你代码里的 Unknown 过滤

# 大洲列表
continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']

# 统计各洲样本数
continent_counts = df_filtered['Area'].value_counts()
total = continent_counts.sum()
print(total)

# 颜色定义（低饱和马卡龙）
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
risk_order = ['Low risk', 'Middle risk', 'High risk']

# 绘图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, continent in enumerate(continents):
    if continent not in continent_counts:
        continue

    data = df_filtered[df_filtered['Area'] == continent]
    risk_counts = data['Risk'].value_counts().reindex(risk_order, fill_value=0)

    # 饼图大小与样本数成正比
    scale = np.sqrt(continent_counts[continent] / total)**0.5
    ax = axes[i]

    # 环形图（统一环宽，保证都是甜甜圈）
    wedges,  autotexts = ax.pie(
        risk_counts.values,
        colors=[colors[r] for r in risk_order],
       # autopct='%1.1f%%',
        pctdistance=0.85,
        wedgeprops=dict(width=0.3, edgecolor='white', linewidth=0.3)  # 环宽统一 0.3
    )

    # 中心标注洲名
    ax.text(0, 0, continent, ha='center', va='center', fontsize=11)

    ax.set_xlim(-1/scale, 1/scale)
    ax.set_ylim(-1/scale, 1/scale)
    ax.set_aspect('equal')
    ax.axis('off')

# 图例
legend_elements = [plt.Rectangle((0,0),1,1, fc=colors[r], ec='white', lw=0.3) for r in risk_order]
fig.legend(legend_elements, risk_order, loc='upper right', bbox_to_anchor=(0.99,0.98), frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("continent_risk_donut_pie.svg", dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

print("图表已保存为: continent_risk_donut.png")
print("\n各洲样本数量：")
for c in continents:
    if c in continent_counts:
        print(f"{c}: {continent_counts[c]}")