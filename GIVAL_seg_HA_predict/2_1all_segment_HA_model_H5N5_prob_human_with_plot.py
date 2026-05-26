import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

def plot_pin_bar_chart(csv_path):
    # 读取数据并按 prob_human 降序排序
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='prob_human', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df))  # y轴位置索引

    # 定义颜色映射规则（柔和三色）
    color_map = {
        'high risk': '#E64B35',
        'middle risk': '#3182BD',
        'low risk': '#31A354'
    }
    def get_color(val):
        if val >= 2 / 3:
            return color_map['high risk']
        elif val >= 1 / 3:
            return color_map['middle risk']
        else:
            return color_map['low risk']

    df['color'] = df['prob_human'].apply(get_color)

    # 创建画布，高度自适应数据量
    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.35)))

    # 绘制大头针的"针杆"：水平细线
    ax.hlines(y=y_pos, xmin=0, xmax=df['prob_human'],
              color='lightgray', linewidth=1.5, alpha=0.8)
    ax.margins(y=0.01)
    # 绘制大头针的"针头"：彩色圆点
    pin_head_size = 100
    ax.scatter(df['prob_human'], y_pos, s=pin_head_size,
               c=df['color'], edgecolor='none', zorder=3)

    # 图表美化
    ax.set_xlabel('Risk score', fontsize=14)
    ax.set_xlim(0.2, max(df['prob_human']) * 1.1)
    ax.set_ylim(-1, len(df))  # 上下留空
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['seq_id'])

    # y轴标签旋转0度，右对齐防遮挡
    plt.yticks(rotation=0, ha='right', fontsize=11)
    plt.xticks(fontsize=11)

    # 移除所有边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 浅色x轴网格线，增强可读性
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.invert_yaxis()  # 保持降序显示

    # ------------------- 添加自定义图例 -------------------
    legend_elements = [
        plt.scatter([], [], s=pin_head_size, color=color_map['high risk'], label='High risk', edgecolor='none'),
        plt.scatter([], [], s=pin_head_size, color=color_map['middle risk'], label='Middle risk', edgecolor='none'),
        plt.scatter([], [], s=pin_head_size, color=color_map['low risk'], label='Low risk', edgecolor='none')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig('segment_HA_model_H5N5_all_prob_human_risk.pdf',dpi=600,bbox_inches='tight')
    plt.show()

# 运行函数 - 替换为你的CSV路径
if __name__ == '__main__':
    csv_file_path = "test_resnet34_0619_all_testHA_for_predict_model_H5N5_RBD_segment_model.csv"
    plot_pin_bar_chart(csv_file_path)