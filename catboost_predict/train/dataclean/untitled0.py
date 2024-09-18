import pandas as pd
import re

# 加载Excel文件
file_path = 'origin.xlsx'  # 请将此路径替换为您的实际文件路径
df = pd.read_excel(file_path)  # 不指定 sheet_name

# 1. 清理四个Molar Ratio列，移除“mol”单位，并转换为浮点数
molar_ratio_columns = ['Molar Ratio', 'Molar Ratio.1', 'Molar Ratio.2', 'Molar Ratio.3']
for col in molar_ratio_columns:
    df[col] = df[col].astype(str)  # 确保所有值为字符串类型
    df[col] = df[col].str.replace(' mol', '', regex=False)
    df[col] = df[col].replace('nan', pd.NA)  # 将字符串'nan'替换为实际的NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')  # 将列转换为浮点数，同时忽略无法转换的值

# 2. 将“Methods”列中的内容整理为“一步法”或“两步法”
df['Methods'] = df['Methods'].apply(lambda x: 'Two-step' if 'Two-step' in x else ('One-step' if 'One-step' in x else x))

# 3. 处理“Thickness (µm)”列中的范围值，取中间值
def calculate_midpoint(value):
    if isinstance(value, str) and '-' in value:
        parts = value.split('-')
        if len(parts) == 2:
            try:
                low = float(parts[0])
                high = float(parts[1])
                return (low + high) / 2
            except ValueError:
                return pd.NA
    try:
        return float(value)
    except (ValueError, TypeError):
        return pd.NA

df['Thickness (µm)'] = df['Thickness (µm)'].apply(calculate_midpoint)

# 4. 清理后面五个性能列
performance_columns = ['Tensile Strength (MPa)', 'Tensile Modulus (GPa)', 'T5% (℃)', 'CTE (ppm/K)', 'Tg (℃)']

def clean_performance_value(value):
    if isinstance(value, str):
        if '±' in value:
            try:
                base_value = float(value.split('±')[0])
                return base_value
            except ValueError:
                return pd.NA
        elif '/' in value:
            try:
                values = [float(v) for v in value.split('/')]
                return max(values)
            except ValueError:
                return pd.NA
        else:
            try:
                return float(value)
            except ValueError:
                return pd.NA
    elif isinstance(value, (int, float)):
        return value
    else:
        return pd.NA

for col in performance_columns:
    df[col] = df[col].apply(clean_performance_value)

# 将清洗后的数据保存为新的Excel文件
output_file_path = 'cleaned_origin1.xlsx'  # 请将此路径替换为您希望保存的路径
df.to_excel(output_file_path, index=False)

print(f"数据清洗完成。清洗后的数据已保存至 {output_file_path}")
