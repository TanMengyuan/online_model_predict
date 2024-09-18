import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = 'TS.csv'  # 请根据实际路径修改
data = pd.read_csv(file_path)

# 计算第一四分位数和第三四分位数
Q1 = data['targets'].quantile(0.25)
Q3 = data['targets'].quantile(0.75)
IQR = Q3 - Q1

# 确定异常值的阈值
upper_threshold = Q3 + 1.5 * IQR
lower_threshold = Q1 - 1.5 * IQR

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

# 输出异常值的信息
print("异常值的信息：")
print(outliers)

# 将异常值数据保存到CSV文件
outliers_path = 'TS-TQR.csv'  # 你可以根据需要修改文件名和路径
outliers.to_csv(outliers_path, index=False)

print(f"潜在需要删除的样本编号: {outliers['number'].tolist()}")
print(f"异常值已保存到文件: {outliers_path}")
