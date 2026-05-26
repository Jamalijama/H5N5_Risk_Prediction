import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 1. 配置参数与绘图风格 --------------------------
input_csv_path = "test_500+0.03_10_resnet34_reassort_H5N5_all_H3N2ref_0.05_0701_2816_fc.csv"
output_counts_csv = "gene_combination_risk_counts_H5N5_H3.csv"

# 莫兰迪和谐配色（按风险等级区分主色调，同风险内基因组合用同色系渐变）
risk_color_map = {
    "Low risk": ["#82B366", "#96C77E", "#AAD197", "#BEDBAF", "#D2E5C7"],
    "Middle risk": ["#6C8EBF", "#829FCF", "#99B2DF", "#AFC5EF", "#C5D8FF"],
    "High risk": ["#B85450", "#C66A66", "#D4807C", "#E29692", "#F0ACA8"]
}
# 预设基因组合列表
all_combinations = [
    "PB2", "PB1", "PA", "NP",
    "PB2_PB1", "PB2_PA", "PB2_NP",
    "PB1_PA", "PB1_NP", "PA_NP"
]

# 设置学术绘图风格
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 0.8
sns.set_style("white")


# -------------------------- 2. 读取数据 + 基因组合提取 + 风险等级划分 --------------------------
def extract_gene_combination(id_string):
    if pd.isna(id_string):
        return "No_replacement"
    match = re.search(r'replace_(.*)', id_string)
    return match.group(1) if match else "No_replacement"


def classify_risk(prob):
    if prob < 1 / 3:
        return "Low risk"
    elif prob < 2 / 3:
        return "Middle risk"
    else:
        return "High risk"


# 数据加载与预处理
data = pd.read_csv(input_csv_path)
data['gene_combination'] = data['id'].apply(extract_gene_combination)
data['risk_level'] = data['prob'].apply(classify_risk)

# 筛选出目标基因组合的数据
data_target = data[data['gene_combination'].isin(all_combinations)].copy()

# -------------------------- 3. 按风险等级统计基因组合计数 --------------------------
risk_gene_stats = []
for risk in ["Low risk", "Middle risk", "High risk"]:
    risk_data = data_target[data_target['risk_level'] == risk]
    gene_count = risk_data.groupby('gene_combination').size().reindex(all_combinations, fill_value=0).reset_index()
    gene_count.columns = ['gene_combination', 'count']
    gene_count['risk_level'] = risk
    risk_gene_stats.append(gene_count)

risk_gene_df = pd.concat(risk_gene_stats, ignore_index=True)

# -------------------------- 4. 绘制分组柱状图（一行三列布局） --------------------------
fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
risk_order = ["Low risk", "Middle risk", "High risk"]

for idx, risk in enumerate(risk_order):
    ax = axes[idx]
    # 提取当前风险等级的基因计数
    plot_data = risk_gene_df[risk_gene_df['risk_level'] == risk]
    # 获取对应配色（循环使用以覆盖所有基因组合）
    colors = risk_color_map[risk] * (len(all_combinations) // len(risk_color_map[risk]) + 1)
    colors = colors[:len(all_combinations)]

    # 绘制柱状图
    bars = ax.bar(
        x=plot_data['gene_combination'],
        height=plot_data['count'],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.7
    )

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                f"{int(height)}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333333"
            )

    # 设置子图标题与标签
    ax.set_title(f"{risk}", fontsize=14, fontweight="bold", color="#2C3E50", pad=15)
    #ax.set_xlabel("Gene Combination", fontsize=12, fontweight="bold", color="#2C3E50", labelpad=10)
    ax.tick_params(axis='x', rotation=45, labelsize=13, labelcolor="#2C3E50")
    ax.tick_params(axis='y', labelsize=13, labelcolor="#2C3E50")

    # 网格线与边框设置
    ax.grid(axis='y', linestyle="--", alpha=0.3, color="#666666")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#BDC3C7")

# 设置共用Y轴标签
fig.text(0.06, 0.5, "Sample Count", va="center", rotation="vertical", fontsize=14, color="#2C3E50")
# 设置总标题
#fig.suptitle("Gene Combination Counts by Risk Level Group", fontsize=16, fontweight="bold", color="#2C3E50", y=0.98)

# 调整布局避免重叠
plt.tight_layout(rect=[0.08, 0.02, 0.98, 0.92])
# 保存高清图片
plt.savefig("risk_grouped_gene_combination_barplot_H5N5_H3.pdf", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()

# -------------------------- 5. 输出统计结果 --------------------------
risk_gene_df.to_csv(output_counts_csv, index=False)
print("各风险等级下基因组合计数统计：")
print(risk_gene_df.pivot(index="gene_combination", columns="risk_level", values="count").fillna(0).astype(int))
print(f"\n统计结果已保存至: {output_counts_csv}")