import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFE
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time  # 导入时间模块


# 时间自适应显示函数（包括分钟、秒等详细格式）
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes} min {seconds:.2f} sec"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours} hr {minutes} min {seconds:.2f} sec"


# 初始化结果列表
all_results = []

round = 21
# 循环处理20对数据集
for i in range(1, round):
    train_file = f'train{i}.csv'  # 根据实际路径调整
    test_file = f'test{i}.csv'  # 根据实际路径调整

    # 加载训练集和测试集
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train = train_data.iloc[:, 1:]  # 训练集特征
    y_train = train_data.iloc[:, 0]  # 训练集目标变量
    X_test = test_data.iloc[:, 1:]  # 测试集特征
    y_test = test_data.iloc[:, 0]  # 测试集目标变量

    # 初始化CatBoost回归器
    catboost = CatBoostRegressor(verbose=0)  # 设置verbose=0以减少日志输出

    # 特征数量范围
    feature_range = range(20, 4, -1)  # 从20个特征减少到5个特征

    # 存储当前数据集的结果
    results = []

    # 初始化时间记录
    times_per_iteration = []

    for n_features in feature_range:
        print(f"i: {i} | n_features: {n_features} | start")

        # 记录开始时间
        start_time = time.time()

        # RFE进行特征选择
        selector = RFE(estimator=catboost, n_features_to_select=n_features, step=1)
        selector = selector.fit(X_train, y_train)

        # 获取选择的特征
        selected_features = X_train.columns[selector.support_]

        # 基于选择的特征重新训练CatBoost模型
        catboost_model = CatBoostRegressor(verbose=0)
        catboost_model.fit(X_train[selected_features], y_train)

        # 在测试集上评估模型
        y_pred_test = catboost_model.predict(X_test[selected_features])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        # 五折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_rmse = -cross_val_score(catboost_model, X_train[selected_features], y_train,
                                   scoring='neg_root_mean_squared_error', cv=kf).mean()
        cv_r2 = cross_val_score(catboost_model, X_train[selected_features], y_train, scoring='r2', cv=kf).mean()

        # 记录结束时间
        end_time = time.time()
        iteration_time = end_time - start_time
        times_per_iteration.append(iteration_time)

        # 估算剩余时间
        avg_time_per_iteration = np.mean(times_per_iteration)
        remaining_iterations = (len(feature_range) - feature_range.index(n_features) - 1) + (round - i - 1) * len(
            feature_range)
        estimated_time_remaining = avg_time_per_iteration * remaining_iterations

        # 自适应时间显示
        formatted_iteration_time = format_time(iteration_time)
        formatted_remaining_time = format_time(estimated_time_remaining)

        print(
            f"i: {i} | n_features: {n_features} | end | iteration time: {formatted_iteration_time} | estimated remaining time: {formatted_remaining_time}")

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
all_results_df.to_csv('tg-rfecatboost.csv', index=False)
