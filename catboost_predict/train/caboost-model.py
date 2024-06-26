import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# 加载数据集
train_data = pd.read_csv('train15-ts.csv')  # 替换为正确的文件路径
test_data = pd.read_csv('train15-ts.csv')  # 替换为正确的文件路径

# 选定的特征
selected_features = ['A2245273601', 'A2245384272', 'A668520641', 'A269959362', 'A3162837314', 'A994485099',
                     'A2246728737', 'A1543552144', 'A3372254436', 'A2929966416', 'A3217380708', 'A951226070',
                     'G3974328374', 'G994485099']

# 分离特征和目标
X_train = train_data[selected_features]
y_train = train_data.iloc[:, 0]
X_test = test_data[selected_features]
y_test = test_data.iloc[:, 0]

# 定义模型参数
params = {
    'learning_rate': 0.1,
    'depth': 5,
    'subsample': 0.7,
    'iterations': 100,
    'verbose': 0
}

# 初始化CatBoost回归器
catboost = CatBoostRegressor(**params)

# 使用五折交叉验证进行模型评估
kf = KFold(n_splits=5)
train_r2_scores = []
test_r2_scores = []
train_rmse_scores = []
test_rmse_scores = []

for train_index, test_index in kf.split(X_train):
    X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

    catboost.fit(X_train_cv, y_train_cv)

    # 计算训练集和验证集的R2和RMSE
    y_pred_train_cv = catboost.predict(X_train_cv)
    y_pred_test_cv = catboost.predict(X_test_cv)
    train_rmse_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv)))
    test_rmse_scores.append(np.sqrt(mean_squared_error(y_test_cv, y_pred_test_cv)))
    train_r2_scores.append(r2_score(y_train_cv, y_pred_train_cv))
    test_r2_scores.append(r2_score(y_test_cv, y_pred_test_cv))

# 输出平均R2和RMSE分数
print(f"Average Train RMSE: {np.mean(train_rmse_scores)}")
print(f"Average Validation RMSE: {np.mean(test_rmse_scores)}")
print(f"Average Train R2: {np.mean(train_r2_scores)}")
print(f"Average Validation R2: {np.mean(test_r2_scores)}")

# 使用所有训练数据重新训练模型
catboost.fit(X_train, y_train)

# 对训练集和测试集进行预测
y_pred_train = catboost.predict(X_train)
y_pred_test = catboost.predict(X_test)

# 模型保存路径
model_save_path = '../predict_model/catboost_model.cbm'
# 保存模型
catboost.save_model(model_save_path, format='cbm')
print(f"Model has been saved to {model_save_path}")

# # 保存预测结果
train_predictions = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})

train_predictions.to_csv('./out/train_predictions-ts-tt15-14.csv', index=False)
test_predictions.to_csv('./out/test_predictions-ts-tt15-14.csv', index=False)

print("Train and test predictions have been saved.")
