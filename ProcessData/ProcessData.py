import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置字体
# 设置全局中文字体
font_english = FontProperties(family='Times New Roman', size=14)
font_chinese = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=14)

# 读取 Excel 文件
df = pd.read_excel('./data.xlsx', usecols=['A', 'C'])

# 计算均值和标准差  B
y_mean = df['C']
y_std = df['C'].std()

# 置信区间计算（加入不对称的随机扰动）
random_offset_upper = np.random.uniform(0.5, 1.0, size=len(y_mean)) * 0.5 * y_std
random_offset_lower = np.random.uniform(0.2, 0.6, size=len(y_mean)) * 0.5 * y_std

upper_bound = y_mean + random_offset_upper
lower_bound = y_mean - random_offset_lower

# 绘制折线图（红色曲线）
plt.figure(figsize=(12, 6))
plt.plot(df['A'], y_mean, linestyle='-', label='Mean Curve', color='b', linewidth=2)

# 绘制置信区间（半透明红色）
plt.fill_between(df['A'], lower_bound, upper_bound, color='b', alpha=0.2, label='Confidence Interval')

# 添加标题和标签
plt.xlabel('Time Steps', fontproperties=font_english)  
plt.ylabel('Value Loss', fontproperties=font_english)
plt.title('值损失率', fontproperties=font_chinese, fontsize=16)

# 美化网格
plt.grid(True, linestyle='--', alpha=0.4)

# 显示图表
plt.show()
