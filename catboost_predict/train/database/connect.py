import pandas as pd

# 加载CSV文件
a_connect = pd.read_csv('g-connect.csv')
mff_connect = pd.read_csv('g-mff-connect.csv')

# 将mff-connect的index列设置为索引列
mff_connect.set_index(mff_connect.columns[0], inplace=True)

# 确保mff_connect的所有特征列都是数值类型
mff_connect = mff_connect.apply(pd.to_numeric, errors='coerce')

# 定义一个函数来计算每一行的特征值加权和
def calculate_features(row, mff_connect):
    features = pd.Series(0, index=mff_connect.columns, dtype=float)
    
    # 处理Diamine 1
    if not pd.isna(row['Diamine 1']):
        diamine_1 = int(row['Diamine 1'])
        molar_ratio_1 = row['Molar Ratio3']
        if diamine_1 in mff_connect.index:
            features += mff_connect.loc[diamine_1] * molar_ratio_1
    
    # 处理Diamine 2（如果存在）
    if not pd.isna(row['Diamine 2']):
        diamine_2 = int(row['Diamine 2'])
        molar_ratio_2 = row['Molar Ratio4']
        if diamine_2 in mff_connect.index:
            features += mff_connect.loc[diamine_2] * molar_ratio_2
    
    return features

# 对a_connect中的每一行应用上述函数
result_df = a_connect.apply(lambda row: calculate_features(row, mff_connect), axis=1)

# 将结果保存到一个新的CSV文件中
result_df.to_csv('g-result.csv', index=False)

print("结果已保存到 'result.csv'.")
