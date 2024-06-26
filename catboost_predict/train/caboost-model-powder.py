import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 加载数据集
train_data = pd.read_csv('train4.csv')  # 替换为正确的文件路径
test_data = pd.read_csv('test4.csv')  # 替换为正确的文件路径

# 选定的特征
selected_features = ['A4216335232-1', 'A951226070-2', 'A2720313463-2', 'A2041434490-2', 'A2084364935-2',
                     'G3692055567-2', 'G3276511768-2']

# 分离特征和目标
X_train = train_data[selected_features]
y_train = train_data.iloc[:, 0]
X_test = test_data[selected_features]
y_test = test_data.iloc[:, 0]

# 定义模型参数
params = {
    'learning_rate': 0.3,
    'depth': 7,
    'subsample': 0.9,
    'iterations': 100,
    'verbose': 0
}

# 初始化CatBoost回归器
catboost = CatBoostRegressor(**params)

# 使用五折交叉验证进行模型评估
kf = KFold(n_splits=5)
train_r2_scores = []
train_rmse_scores = []
test_predictions_list = []

for train_index, test_index in kf.split(X_train):
    X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

    catboost.fit(X_train_cv, y_train_cv)

    # 计算训练集的R2和RMSE
    y_pred_train_cv = catboost.predict(X_train_cv)
    train_rmse_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv)))
    train_r2_scores.append(r2_score(y_train_cv, y_pred_train_cv))

    # 对测试集进行预测并保存预测结果
    y_pred_test_cv = catboost.predict(X_test)
    test_predictions_list.append(y_pred_test_cv)

# 对测试集的预测结果进行平均
avg_test_predictions = np.mean(test_predictions_list, axis=0)

# 计算测试集的R2和RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, avg_test_predictions))
test_r2 = r2_score(y_test, avg_test_predictions)

# 输出平均R2和RMSE分数
print(f"Average Train RMSE: {np.mean(train_rmse_scores)}")
print(f"Average Train R2: {np.mean(train_r2_scores)}")
print(f"Test RMSE: {test_rmse}")
print(f"Test R2: {test_r2}")

# 使用所有训练数据重新训练模型
catboost.fit(X_train, y_train)

# 对训练集进行预测
y_pred_train = catboost.predict(X_train)

# 保存预测结果
train_predictions = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': avg_test_predictions})

# 模型保存路径
model_save_path = '../predict_model/powder/catboost_model.cbm'
# 保存模型
catboost.save_model(model_save_path, format='cbm')
print(f"Model has been saved to {model_save_path}")

# train_predictions.to_csv('train_predictions-tt4-model-C.csv', index=False)
# test_predictions.to_csv('test_predictions-tt4-model-C.csv', index=False)

print("Train and test predictions have been saved.")
