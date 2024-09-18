# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:50:46 2024

@author: 15297
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归器
from sklearn.metrics import mean_squared_error, r2_score

# 加载训练集和测试集
train_data = pd.read_csv('train7.csv')  # 替换为正确的文件路径
test_data = pd.read_csv('test7.csv')    # 替换为正确的文件路径

X_train = train_data.iloc[:, 1:]  # 训练集特征
y_train = train_data.iloc[:, 0]  # 训练集目标变量
X_test = test_data.iloc[:, 1:]   # 测试集特征
y_test = test_data.iloc[:, 0]    # 测试集目标变量

# 初始化随机森林回归器
random_forest = RandomForestRegressor()  # 可以调整n_estimators等参数
#参数n_estimators=100, random_state=42
# 特征数量范围
feature_range = range(20, 4, -1)  # 从20个特征减少到5个特征

# 存储结果和特征子集
results = []
feature_subsets = {}

max_features = max(feature_range)
for n_features in feature_range:
    # RFE进行特征选择
    selector = RFE(random_forest, n_features_to_select=n_features, step=1)
    selector = selector.fit(X_train, y_train)

    # 获取选择的特征
    selected_features = X_train.columns[selector.support_]

    # 保存特征子集
    feature_subsets[f'Features_{n_features}'] = selected_features.tolist() + [None] * (max_features - n_features)

    # 基于选择的特征重新训练随机森林模型
    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(X_train[selected_features], y_train)

    # 在测试集上评估模型
    y_pred_test = random_forest_model.predict(X_test[selected_features])
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    # 五折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rmse = -cross_val_score(random_forest_model, X_train[selected_features], y_train, scoring='neg_root_mean_squared_error', cv=kf).mean()
    cv_r2 = cross_val_score(random_forest_model, X_train[selected_features], y_train, scoring='r2', cv=kf).mean()

    # 保存结果
    results.append({
        'n_features': n_features,
        'Test_RMSE': test_rmse, 
        'Test_R2': test_r2,
        'CV_RMSE': cv_rmse, 
        'CV_R2': cv_r2
    })

# 将特征子集和结果转换为DataFrame
results_df = pd.DataFrame(results)
feature_subsets_df = pd.DataFrame(feature_subsets)

# 输出结果和特征子集到CSV文件
results_df.to_csv('rf-tstt1-result3.csv', index=False)
feature_subsets_df.to_csv('rf-tstt1-feature3.csv', index=False)

# 打印确认
print("Model evaluation and feature subsets have been saved to CSV files.")
