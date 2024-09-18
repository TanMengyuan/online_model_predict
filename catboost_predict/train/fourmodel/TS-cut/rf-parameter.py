# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:54:41 2024

@author: 15297
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归器
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

# 加载训练集和测试集
train_data = pd.read_csv('train1.csv')  # 替换为正确的文件路径
test_data = pd.read_csv('test1.csv')    # 替换为正确的文件路径

# 选定的特征
selected_features = ['copolymer', 'Maximum Imidization Temperature', 'Total Imidization Time', 'Thickness-middle', 'a27979804', 'a4256002933', 'a951226070', 'g3974328374']

# 分离特征和目标
X_train = train_data[selected_features]
y_train = train_data.iloc[:, 0]
X_test = test_data[selected_features]
y_test = test_data.iloc[:, 0]

# 参数网格
# 参数网格
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],  # 将 'auto' 替换为 None
    'bootstrap': [True, False],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_samples': [None, 0.8, 0.9]
}

# 初始化随机森林回归器
rf = RandomForestRegressor()

# 十折交叉验证
kf = KFold(n_splits=10)

# 网格搜索
grid_search = GridSearchCV(rf, param_grid, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train)

# 结果输出
results = []
for i, params in enumerate(grid_search.cv_results_['params']):
    mean_train_rmse = -grid_search.cv_results_['mean_train_score'][i]  # 从交叉验证结果中提取训练集的平均RMSE
    mean_train_r2 = grid_search.cv_results_['mean_train_score'][i]  # 如果需要从交叉验证计算训练集的R²
    
    # 使用最佳参数重新训练模型并计算测试集的评分
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)  # 使用 mean_squared_error 计算 RMSE
    test_r2 = r2_score(y_test, y_pred_test)
    
    # 添加到结果中
    results.append({
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'max_features': params['max_features'],
        'bootstrap': params['bootstrap'],
        'min_weight_fraction_leaf': params['min_weight_fraction_leaf'],
        'max_samples': params.get('max_samples', None),
        'CV-10-train-RMSE': mean_train_rmse,  # 训练集上的RMSE
        'CV-10-train-R2': mean_train_r2,  # 训练集上的R2
        'test-RMSE': test_rmse,
        'test-R2': test_r2
    })

# 保存结果到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv('rf_parameter-tt1.csv', index=False)

print("Parameter optimization results saved to 'rf_parameter_optimization.csv'.")
