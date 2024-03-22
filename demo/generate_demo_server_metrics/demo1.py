# Author:Jane
# -*- coding = utf-8 -*-
# @Time :2024/3/22 19:08
# @Author :langyanping1
# @Site :
# @File :demo1.py
# @software :
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件为DataFrame
df = pd.read_csv('synthetic_48.csv')

# 将时间戳转换为日期时间格式
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# 创建两个子图
fig, ax = plt.subplots()

# 绘制全部数值
ax.plot(df['timestamp'], df['value'], label='Value', color='blue')

# 标记异常值
anomalies = df[df['is_anomaly'] == 1]  # 假设标记为1的是异常值
ax.scatter(anomalies['timestamp'], anomalies['value'], color='red', label='Anomalies')

# 添加图例和标签
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Time Series Data with Anomalies')
ax.legend()

# 展示图表
plt.show()