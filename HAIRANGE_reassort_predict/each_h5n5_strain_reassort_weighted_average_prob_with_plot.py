import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


def plot_reassortment_risk(csv_path):
    # 1. 读取数据
    df = pd.read_csv(csv_path)

    # 2. 提取真实病毒 ID：按 _replace 拆分，取第一段
    def get_real_id(x):
        if '_replace' in x:
            return x.split('_replace')[0]
        return x

    df['real_id'] = df['id'].apply(get_real_id)

    # 3. 判断单/双/三片段：看 replace 后面部分有几个下划线
    def get_type(s):
        if '_replace' not in s:
            return 'unknown'
        tail = s.split('_replace')[1]
        n = tail.count('_')
        if n == 1:
            return 'single'
        elif n == 2:
            return 'double'
        elif n == 3:
            return 'triple'
        else:
            return 'unknown'

    df['type'] = df['id'].apply(get_type)

    # 4. 按每个真实ID计算总分：4*单 +6*双 +4*三
    def calc_score(group):
        s1 = group[group['type'] == 'single']['prob'].mean()
        s2 = group[group['type'] == 'double']['prob'].mean()
        s3 = group[group['type'] == 'triple']['prob'].mean()

        s1 = s1 if not np.isnan(s1) else 0
        s2 = s2 if not np.isnan(s2) else 0
        s3 = s3 if not np.isnan(s3) else 0

        return 4 * s1 + 6 * s2 + 4 * s3

    res = df.groupby('real_id').apply(calc_score).reset_index()
    res.columns = ['seq_id', 'total_score']

    # 5. 按总分降序排序
    res = res.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(res))

    # ===================== 画图（完全沿用你的风格） =====================
    fig, ax = plt.subplots(figsize=(8, max(5, len(res) * 0.35)))

    # 针杆
    ax.hlines(y=y_pos, xmin=0, xmax=res['total_score'],
              color='lightgray', linewidth=1.5, alpha=0.8)
    ax.margins(y=0.01)

    # 针头：统一蓝色，无高低风险
    pin_head_size = 100
    ax.scatter(res['total_score'], y_pos, s=pin_head_size,
               c='#3182BD', edgecolor='none', zorder=3)

    ax.set_xlabel('Total reassortment risk score', fontsize=14)
    ax.set_xlim(0, max(res['total_score']) * 1.1)
    ax.set_ylim(-1, len(res))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(res['seq_id'])
    plt.yticks(rotation=0, ha='right', fontsize=11)
    plt.xticks(fontsize=11)

    # 去掉所有边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('reassortment_risk_result_each_H5N5_weighted_average.pdf', dpi=600, bbox_inches='tight')
    plt.show()

    # 保存结果
    res.to_csv('reassortment_risk_result_each_H5N5_weighted_average.csv', index=False)


# ===================== 在这里改你的文件路径 =====================
if __name__ == '__main__':
    csv_file_path = "test_500+0.03_10_resnet34_reassort_H5N5_all_H3N2ref_0.05_0701_2816_fc.csv"
    plot_reassortment_risk(csv_file_path)