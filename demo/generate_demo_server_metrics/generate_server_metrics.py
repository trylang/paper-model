# Author:Jane
# -*- coding = utf-8 -*-
# @Time :2024/3/22 18:58
# @Author :langyanping1
# @Site :
# @File :generate_server_metrics.py
# @software :

import pandas as pd
import random
from datetime import datetime, timedelta

# 生成时间戳
start_time = datetime(2022, 1, 1)
end_time = datetime(2022, 1, 31)
time_stamps = [start_time + timedelta(hours=i) for i in range(24*31)]  # 生成 31 天的数据，每小时一个数据点

# 生成 CPU 使用率数据
cpu_usage = [random.randint(1, 100) for _ in range(24*31)]

# 加入异常点数据
# 假设异常点是 CPU 使用率超过 90% 的数据点
num_anomalies = 10  # 设置异常点数量
anomaly_indices = random.sample(range(0, len(cpu_usage)), num_anomalies)  # 随机选择异常点的索引
for idx in anomaly_indices:
    cpu_usage[idx] = random.randint(91, 100)  # 将异常点的 CPU 使用率设置为大于 90%

# 创建 pandas DataFrame
data = pd.DataFrame({'timestamp': time_stamps, 'cpu_usage': cpu_usage})

# 将数据写入 CSV 文件
data.to_csv('server_metrics.csv', index=False)
