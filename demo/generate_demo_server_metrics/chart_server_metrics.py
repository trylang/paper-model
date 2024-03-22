# Author:Jane
# -*- coding = utf-8 -*-
# @Time :2024/3/22 19:01
# @Author :langyanping1
# @Site :
# @File :chart_server_metrics.py
# @software :
import pandas as pd
import matplotlib.pyplot as plt

# 假设服务器指标数据存储在名为 server_metrics.csv 的 CSV 文件中，包括时间戳和 CPU 使用率
data = pd.read_csv('synthetic_48.csv')

# 将时间戳列转换为日期时间类型
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 绘制 CPU 使用率时序图
plt.figure(figsize=(15, 6))
plt.plot(data['timestamp'], data['value'], label='CPU Usage', color='blue')

# 标记异常点
# 这里假设异常点是 CPU 使用率超过阈值的数据点
threshold = 80  # 设置异常点阈值为 80%
anomalies = data[data['value'] > threshold]
plt.scatter(anomalies['timestamp'], anomalies['value'], color='red', label='Anomalies')

plt.title('Server CPU Usage Over Time')
plt.xlabel('Time')
plt.ylabel('CPU Usage (%)')
plt.legend()
plt.show()