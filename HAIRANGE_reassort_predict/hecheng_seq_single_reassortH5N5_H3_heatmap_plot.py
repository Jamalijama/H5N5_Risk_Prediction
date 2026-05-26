import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 1. 配置参数 --------------------------
input_csv_path = "test_500+0.03_10_resnet34_reassort_H5N5_all_H3N2ref_0.05_0701_2816_fc.csv"
output_csv_path = "reassort_risk_predict_for_8_hecheng_strains.csv"
original_8_sequences = [
    'A/Washington/2148/2025',
    'A/mallard/California/1400/2013',
    'A/mallard/California/1376/2013',
    'A/mallard/Minnesota/355811/2000',
    'A/bald_eagle/Ohio/OH25-5320/2025',
    'A/chicken/CO/24-036222-001-original-repeat/2024',
    'A/purple heron/Egypt/MB 933C/2016',
    'A/mallard/Minnesota/16-041335-4/2016'
]
suffixes = ["_replace_PB2", "_replace_PB1", "_replace_PA", "_replace_NP"]
segments = ["PB2", "PB1", "PA", "NP"]
colors = ["#4198AC", "#7BC0CD", "#ED8D5A", "#ECB66C"]  # 莫兰迪色系

# -------------------------- 2. 生成32个目标ID并提取样本 --------------------------
target_ids = [f"{seq}{suffix}" for seq in original_8_sequences for suffix in suffixes]
df = pd.read_csv(input_csv_path)
df_extracted = df[df["id"].isin(target_ids)].copy()
df_extracted = df_extracted.sort_values(by="prob", ascending=True).reset_index(drop=True)
df_extracted.to_csv(output_csv_path, index=False)
print(f"已提取并排序{len(df_extracted)}条样本，保存到: {output_csv_path}")


# -------------------------- 3. 风险分类 + 提取片段类型 --------------------------
def classify_risk(prob):
    if prob < 1 / 3:
        return "Low risk"
    elif prob < 2 / 3:
        return "Middle risk"
    else:
        return "High risk"


df_extracted["risk_level"] = df_extracted["prob"].apply(classify_risk)


def get_segment(id_str):
    for s in suffixes:
        if s in id_str:
            return s.replace("_replace_", "")


df_extracted["segment"] = df_extracted["id"].apply(get_segment)

# -------------------------- 4. 统计各风险等级下的片段占比 --------------------------
risk_order = ["High risk", "Middle risk", "Low risk"]
segment_counts = {}
for risk in risk_order:
    count = df_extracted[df_extracted["risk_level"] == risk]["segment"].value_counts()
    segment_counts[risk] = [count.get(seg, 0) for seg in segments]

# -------------------------- 5. 绘图布局：3环形图 + 热图（无报错版） --------------------------
plt.rcParams["font.family"] = "Arial"
# 使用GridSpec精准划分布局
fig = plt.figure(figsize=(34, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[0.7, 0.9], hspace=0.15, wspace=0.02)

# --- 绘制3个环形图（上排3个位置）---
ax_pie1 = fig.add_subplot(gs[0, 0])  # High risk 环形图
ax_pie2 = fig.add_subplot(gs[0, 1])  # Middle risk 环形图
ax_pie3 = fig.add_subplot(gs[0, 2])  # Low risk 环形图
pie_axes = [ax_pie1, ax_pie2, ax_pie3]
# 调整三个环形图的位置：都向左移动并缩小间距
# 热图位置保持不变
pie_positions = [
    [0.15, 0.65, 0.15, 0.15],  # High risk: [left, bottom, width, height]
    [0.25, 0.65, 0.15, 0.15],  # Middle risk
    [0.45, 0.65, 0.15, 0.15]   # Low risk
]

for ax, pos in zip(pie_axes, pie_positions):
    ax.set_position(pos)

# 将热图固定在下方的固定位置，不受环形图调整影响
# 热图的GridSpec位置是 gs[1, :]，已经确定




for ax, risk in zip(pie_axes, risk_order):
    counts = segment_counts[risk]
    # 过滤掉数量为0的片段
    valid_counts = [c for c in counts if c > 0]
    valid_colors = [colors[i] for i, cnt in enumerate(counts) if cnt > 0]

    if valid_counts:
        # 绘制环形图：wedgeprops设置圆环宽度
        wedges, texts = ax.pie(
            valid_counts,
            labels=None,
            colors=valid_colors,
            # autopct="%1.1f%%",
            # autopct="none",
            startangle=90,
            textprops={"fontsize": 10},
            #wedgeprops={"edgecolor": "black", "linewidth": 0.2, "width": 0.4}
            wedgeprops={"width": 0.4}
        )
        # 调整百分比文字颜色
        # for autotext in autotexts:
        #     autotext.set_color("white")
    ax.axis("equal")

    # 在环形图中间添加风险等级文字
    ax.text(
        0, 0, risk,
        ha="center", va="center",
        fontsize=14
    )

# --- 绘制热图（下排合并3列）---
ax_heatmap = fig.add_subplot(gs[1, :])
# 构建热图矩阵
heatmap_data = pd.DataFrame(
    index=risk_order,
    columns=df_extracted["id"].tolist(),
    data=np.nan
)
for _, row in df_extracted.iterrows():
    heatmap_data.loc[row["risk_level"], row["id"]] = row["prob"]

# 绘制热图（移除所有无效参数）
sns.heatmap(
    heatmap_data,
    ax=ax_heatmap,
    cmap="Reds",
    vmin=0,
    vmax=1,
    annot=False,
    cbar_kws={
        "shrink": 0.4,
        "pad": 0.01,
        "orientation": "vertical",
        "location": "right"
    },
    linewidths=0.3,
    linecolor="lightgray",
    mask=np.isnan(heatmap_data),
    square=False
)

# 设置热图黑色边框
for spine in ax_heatmap.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1)

# 调整热图刻度（无报错写法）
ax_heatmap.tick_params(axis="x", labelsize=10, rotation=45)
ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), ha="right")
ax_heatmap.tick_params(axis="y", labelsize=12, rotation=0)

# 设置色条文字大小（正确方法）
cbar = ax_heatmap.collections[0].colorbar
cbar.set_label('Probability', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# 添加统一图例
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=c, label=s) for s, c in zip(segments, colors)]
fig.legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(0.99, 0.95),
    fontsize=11,
    frameon=True,
    edgecolor="black"
)

# 调整整体边距，防止遮挡
plt.subplots_adjust(left=0.2, right=0.99, top=0.94, bottom=0.4)

# 保存高清图片
plt.savefig("donut_heatmap_combined_final.png", dpi=300, bbox_inches="tight")
plt.show()