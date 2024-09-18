import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据集
data = pd.read_csv('TS-pre-cut.csv')  # 替换为您的文件路径

# 目标列和特征列
y = data.iloc[:, 0]  # 第一列是目标
X = data.iloc[:, 1:]  # 第二列及以后是特征

# 进行20次随机划分
for i in range(1, 21):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=i)

    # 合并目标和特征列
    train_data = pd.concat([y_train, X_train], axis=1)
    test_data = pd.concat([y_test, X_test], axis=1)

    # 保存为CSV文件
    train_data.to_csv(f'train{i}.csv', index=False)
    test_data.to_csv(f'test{i}.csv', index=False)
