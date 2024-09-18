import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = 'TS.csv'  # 请根据实际路径修改
data = pd.read_csv(file_path)

# 计算均值和标准差
mean_targets = data['targets'].mean()
std_targets = data['targets'].std()

# 确定异常值的阈值
upper_threshold = mean_targets + 3 * std_targets
lower_threshold = mean_targets - 3 * std_targets

# 识别异常值
outliers = data[(data['targets'] > upper_threshold) | (data['targets'] < lower_threshold)]

# 可视化 'targets' 列来标识异常值
plt.figure(figsize=(10, 6))
sns.boxplot(data['targets'])
plt.title('Box plot of Targets with Outliers Highlighted')

# 标记异常值
for outlier in outliers['targets']:
    plt.annotate('Outlier', xy=(0, outlier), xytext=(0.3, outlier),
                 arrowprops=dict(facecolor='red', shrink=0.05))

plt.show()

# 将异常值数据保存到CSV文件
outliers_path = 'TS-std.csv'  # 你可以根据需要修改文件名和路径
outliers.to_csv(outliers_path, index=False)

print(f"潜在需要删除的样本编号: {outliers['number'].tolist()}")
print(f"异常值已保存到文件: {outliers_path}")
