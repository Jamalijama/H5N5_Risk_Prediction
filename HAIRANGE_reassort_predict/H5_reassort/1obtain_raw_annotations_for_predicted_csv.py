import pandas as pd

# 读取预测结果CSV
df_pred = pd.read_csv("test_500+0.03_10_resnet34_reassort_H5_all_H3N2ref_0.05_0701_2816_fc.csv")  # 替换为你的预测结果文件路径

# 读取原始信息CSV
df_info = pd.read_csv("../../non_h5n1_cleaned_with_year_period.csv")  # 替换为你的原始信息文件路径

# 创建字典：ID -> Host1, Year0, Continent
host_dict = dict(zip(df_info['strain_name'], df_info['Host1']))
year_dict = dict(zip(df_info['strain_name'], df_info['Year0']))
continent_dict = dict(zip(df_info['strain_name'], df_info['Continent']))
sero_dict = dict(zip(df_info['strain_name'], df_info['Serotype_N']))


# 准备新数据的列表
new_data = []

# 遍历预测结果的每一行
for i in range(len(df_pred)):
    # 获取当前行的ID
    original_id = df_pred.loc[i, 'id']

    # 按下划线分割
    parts = original_id.split('_replace_')

    # 提取真实ID和重配基因片段
    real_id = parts[0]
    gene_segment = parts[1] if len(parts) > 1 else ''

    # 查询对应的宿主、年份、大洲
    host = host_dict.get(real_id, 'UNK')
    year = year_dict.get(real_id, 'UNK')
    continent = continent_dict.get(real_id, 'UNK')
    sero = sero_dict.get(real_id, 'UNK')

    # 添加到新数据列表
    new_data.append([gene_segment, host, year, continent,sero])

# 创建新的DataFrame
df_new = pd.DataFrame(new_data, columns=['reassort_seg', 'Host', 'Year', 'Area','Sero_H'])
df_new = pd.concat([df_pred,df_new],axis=1)
# 新增：根据prob列生成Risk列
df_new['Risk'] = df_new['prob'].apply(lambda x:
    'Low risk' if x < 1/3
    else ('Middle risk' if x < 2/3
          else 'High risk')
)
# 保存新表格
df_new.to_csv("annotated_predict_result_non_H5N1_H5N5_H5.csv", index=False)
