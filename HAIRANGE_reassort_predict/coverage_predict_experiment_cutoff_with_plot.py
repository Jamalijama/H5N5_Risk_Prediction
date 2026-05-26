import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 1. 设置全局绘图参数 --------------------------
plt.rcParams['font.family'] = 'Arial'  # 指定字体为Arial
plt.rcParams['font.size'] = 12  # 基础字号
plt.rcParams['axes.linewidth'] = 0.5  # 坐标轴边框宽度
plt.rcParams['axes.spines.top'] = False  # 关闭顶部边框
plt.rcParams['axes.spines.right'] = False  # 关闭右侧边框
plt.rcParams['xtick.major.width'] = 0.5  # x轴刻度线宽度
plt.rcParams['ytick.major.width'] = 0.5  # y轴刻度线宽度
plt.rcParams['figure.dpi'] = 600  # 高清分辨率

# -------------------------- 2. 读取数据并提取adaptive的id列表 --------------------------
file1_path = "experimental_adaptive_8strains_reassort.csv"
file2_path = "test_500+0.03_10_resnet34_reassort_H5N5_all_H3N2ref_0.05_0701_2816_fc.csv"

df_exp = pd.read_csv(file1_path)
adaptive_ids = df_exp[df_exp['result'] == 'adaptive']['id'].tolist()
total_adaptive = len(adaptive_ids)
assert total_adaptive > 0, "Error: 没有筛选到result为adaptive的行"

df_probe = pd.read_csv(file2_path)
assert 'prob' in df_probe.columns, "Error: 第二个csv中没有probe列"
assert 'id' in df_probe.columns, "Error: 第二个csv中没有id列"

# -------------------------- 3. 定义阈值列表并计算各阈值下的覆盖率 --------------------------
cutoffs = np.arange(0, 1.05, 0.05)
coverage_rates = []

for cutoff in cutoffs:
    selected_ids = df_probe[df_probe['prob'] > cutoff]['id'].tolist()
    overlap = len(set(selected_ids) & set(adaptive_ids))
    coverage = (overlap / total_adaptive) * 100
    coverage_rates.append(coverage)

# -------------------------- 4. 绘制覆盖率曲线（X轴倒序：1.0 → 0） --------------------------
fig, ax = plt.subplots(figsize=(30, 6))

# 绘图（数据不变，只反转x轴显示）
ax.plot(cutoffs, coverage_rates, color='#2E86AB', linewidth=1, marker='o',
        markersize=8, markerfacecolor='#A23B72', markeredgecolor='white',
        markeredgewidth=0.5, label=f'Total adaptive samples: {total_adaptive}')

# -------------------------- 5. 设置坐标轴标签与范围（关键：反转x轴） --------------------------
ax.set_xlabel('Probability Threshold Cutoff', fontsize=14, labelpad=10)
ax.set_ylabel('Coverage Rate of Adaptive Samples (%)', fontsize=14, labelpad=10)

# 👇 这一行让 X 轴从 1.0 到 0 倒序
ax.set_xlim(1.05, -0.05)

ax.set_ylim(0, max(coverage_rates) * 1.1)

# 设置x轴刻度
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_xticks(cutoffs, minor=True)
ax.tick_params(axis='x', labelsize=10, rotation=45, pad=5)
ax.tick_params(axis='y', labelsize=10)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.3, color='gray', which='major')
ax.grid(True, linestyle=':', alpha=0.2, color='gray', which='minor')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)

# -------------------------- 6. 保存与显示图片 --------------------------
plt.savefig('prob_experiment_threshold_coverage_curve_reversed.pdf',
            format='pdf', dpi=600, bbox_inches='tight')

# -------------------------- 7. 输出统计信息（不变） --------------------------
print("=" * 60)
print("阈值覆盖率分析报告")
print("=" * 60)
print(f"Adaptive样本总数: {total_adaptive}")
print(f"阈值范围: 0.0 到 1.0 (步长: 0.05)")
print("\n关键阈值点的覆盖率:")
print("-" * 40)

key_cutoffs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for cutoff in key_cutoffs:
    idx = int(cutoff * 20)
    if idx < len(coverage_rates):
        print(f"阈值 {cutoff:.1f}: {coverage_rates[idx]:.2f}%")

max_drop = 0
max_drop_idx = 0
for i in range(1, len(coverage_rates)):
    drop = coverage_rates[i-1] - coverage_rates[i]
    if drop > max_drop:
        max_drop = drop
        max_drop_idx = i

print(f"\n最大单次下降: {max_drop:.2f}% (从阈值 {cutoffs[max_drop_idx-1]:.2f} 到 {cutoffs[max_drop_idx]:.2f})")

for i, rate in enumerate(coverage_rates):
    if rate <= 50:
        print(f"覆盖率降至50%以下: 阈值 {cutoffs[i]:.2f}")
        break

print("\n推荐阈值选择:")
print("-" * 40)
best_threshold = None
best_balance = 0

for i, (cutoff, coverage) in enumerate(zip(cutoffs, coverage_rates)):
    if cutoff > 0.2:
        balance = coverage * 0.8 + cutoff * 20
        if balance > best_balance:
            best_balance = balance
            best_threshold = cutoff

if best_threshold is not None:
    best_idx = int(best_threshold * 20)
    best_coverage = coverage_rates[best_idx]
    print(f"推荐阈值: {best_threshold:.2f} (覆盖率: {best_coverage:.1f}%)")

print("=" * 60)