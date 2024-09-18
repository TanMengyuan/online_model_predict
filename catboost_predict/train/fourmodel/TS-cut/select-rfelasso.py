# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:45:42 2024

@author: 15297
"""

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 初始化结果列表
all_results = []

# 循环处理20对数据集
for i in range(1, 21):
    train_file = f'train{i}.csv'  # 根据实际路径调整
    test_file = f'test{i}.csv'    # 根据实际路径调整
    
    # 加载训练集和测试集
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    X_train = train_data.iloc[:, 1:]  # 训练集特征
    y_train = train_data.iloc[:, 0]   # 训练集目标变量
    X_test = test_data.iloc[:, 1:]    # 测试集特征
    y_test = test_data.iloc[:, 0]     # 测试集目标变量
    
    # 初始化Lasso回归器
    lasso = Lasso()

    # 特征数量范围
    feature_range = range(20, 4, -1)  # 从20个特征减少到5个特征

    # 存储当前数据集的结果
    results = []

    for n_features in feature_range:
        print("i: %s | n_features : %s | start"%(i, n_features))
    
    for n_features in feature_range:
        # RFE进行特征选择
        selector = RFE(estimator=lasso, n_features_to_select=n_features, step=1)
        selector = selector.fit(X_train, y_train)

        # 获取选择的特征
        selected_features = X_train.columns[selector.support_]

        # 基于选择的特征重新训练Lasso模型
        lasso_model = Lasso()
        lasso_model.fit(X_train[selected_features], y_train)

        # 在测试集上评估模型
        y_pred_test = lasso_model.predict(X_test[selected_features])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        # 五折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmse = -cross_val_score(lasso_model, X_train[selected_features], y_train, scoring='neg_root_mean_squared_error', cv=kf).mean()
        cv_r2 = cross_val_score(lasso_model, X_train[selected_features], y_train, scoring='r2', cv=kf).mean()

        # 保存当前迭代的结果
        results.append({
            'Dataset': i,
            'n_features': n_features,
            'CV_RMSE': cv_rmse, 
            'CV_R2': cv_r2,
            'Test_RMSE': test_rmse, 
            'Test_R2': test_r2
        })
    
    # 将当前数据集的结果添加到总结果列表中
    all_results.extend(results)

# 将所有结果转换为DataFrame并打印
all_results_df = pd.DataFrame(all_results)
print(all_results_df)

# 可以选择将结果保存为CSV文件
all_results_df.to_csv('ts-rfelasso.csv', index=False)
