# Author:Jane
# -*- coding = utf-8 -*-
# @Time :2024/3/5 13:40
# @Author :langyanping1
# @Site :
# @File :top15_indicator.py
# @software :

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# 指定默认风格
plt.style.use('fivethirtyeight')

from undersample_value import under_sample_split

# 导入算法
from sklearn.ensemble import RandomForestRegressor
# 导入所需工具包
from sklearn.tree import export_graphviz
import pydot



# 下采样数据
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = under_sample_split()

print(X_train_undersample.head())

# 建模
rf = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)

# 训练
rf.fit(X_train_undersample, y_train_undersample)


##################### 画树图 +++++++++++++++++++++++++++
# 提取一颗树
tree_small = rf.estimators_[5]

# # 保存
# rf.estimators_ 是一个包含所有决策树的列表。通过 rf.estimators_[5] 您尝试获取第六棵树（索引从0开始）。如果您的随机森林只有10棵树（n_estimators=10），那么索引5是有效的。
# feature_names 应该是一个包含特征名称的列表。在您的代码中，X_train_undersample 是一个DataFrame，您应该使用 X_train_undersample.columns 来获取特征名称列表。
# export_graphviz 函数没有问题，但是确保 out_file 指向一个有效的路径，并且您有权限在该路径下创建文件。
# export_graphviz(tree_small, out_file = '../images/small_tree.dot', feature_names = X_train_undersample.columns, rounded = True, precision = 1)
#
# (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
#
# graph.write_png('../images/small_tree.png');


##################### 测试 得 MAPE  +++++++++++++++++++++++
# 预测结果
predictions = rf.predict(X_test_undersample)

# print(33, np.shape(predictions))   # 返回(72,)
# print(44, np.shape(y_test_undersample))  # 返回(72, 1)
# predictions 的形状是 (72,)，这意味着它是一个长度为 72 的一维数组。而 y_test_undersample 的形状是 (72, 1)，这意味着它是一个二维数组，其中包含 72 个元素，每个元素都是一个长度为 1 的子数组。

# mean absolute percentage error (MAPE)
# 问题在于在计算 mape 时，你需要计算的是每个误差占真实值的百分比，所以你应该将每个误差除以对应的真实值，并将结果乘以100来转换为百分比。但是，你不能直接将整个 errors 数组除以 y_test_undersample 数组的单个值（如果你这样做，它将尝试广播，并且可能会给出错误的结果）。
# 请注意，如果 y_test_undersample 中的值非常重要，您可能不希望使用 flatten() 方法，因为这会创建一个数据的副本。在这种情况下，您可以使用 values 属性和 ravel() 方法
# ravel() 方法与 flatten() 类似，但通常更高效，因为它返回的是视图而不是数据的副本（尽管在这种情况下，由于 values 属性返回的是 NumPy 数组，ravel() 和 flatten() 都会返回一个副本）。

# 修正后代码
# 计算误差
errors = abs(predictions - y_test_undersample.values.ravel())

# 计算每个误差占真实值的百分比
mape = 100 * (errors / y_test_undersample.values.ravel())

# 计算平均绝对百分比误差 (MAPE)
# 理论上，MAPE 不应该有负值,但因为y_test_undersample会是负值，所以最后mape_mean 也会是负值
mape_mean = mape.mean()

print(22, mape_mean)


#########   特征重要性  ++++++++++++++++++
importances = list(rf.feature_importances_)

# 假设 'V2' 是您的标签列，从特征数据中移除它, X_train_undersamplen 本身就没有V2，不需要去除
# features = X_train_undersample.drop('V2', axis=1)

# 获取特征名称
feature_list = list(X_train_undersample.columns)

# 特征重要性列表
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# 按重要性降序排序
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# 打印特征及其重要性
for pair in feature_importances:
    print('Variable: {:20} Importance: {}'.format(*pair))


# 重要特征打印结果： [5 rows x 29 columns]
# 22 31.98137995686841
# Variable: V7                   Importance: 0.23
# Variable: normAmount           Importance: 0.16
# Variable: V14                  Importance: 0.13
# Variable: V10                  Importance: 0.11
# Variable: V4                   Importance: 0.07
# Variable: V24                  Importance: 0.07
# Variable: V5                   Importance: 0.06
# Variable: V20                  Importance: 0.06
# Variable: V23                  Importance: 0.04
# Variable: V9                   Importance: 0.02
# Variable: V25                  Importance: 0.02
# Variable: V6                   Importance: 0.01
# Variable: V8                   Importance: 0.01
# Variable: V17                  Importance: 0.01
# Variable: V28                  Importance: 0.01
# Variable: V1                   Importance: 0.0
# Variable: V3                   Importance: 0.0
# Variable: V11                  Importance: 0.0
# Variable: V12                  Importance: 0.0
# Variable: V13                  Importance: 0.0
# Variable: V15                  Importance: 0.0
# Variable: V16                  Importance: 0.0
# Variable: V18                  Importance: 0.0
# Variable: V19                  Importance: 0.0
# Variable: V21                  Importance: 0.0
# Variable: V22                  Importance: 0.0
# Variable: V26                  Importance: 0.0
# Variable: V27                  Importance: 0.0
# Variable: Class                Importance: 0.0

#################   绘图TOP15 ++++++++++++++++++++++

# 假设 feature_importances 是一个包含特征名和对应重要性的列表
# feature_importances = [('feature1', 0.2), ('feature2', 0.5), ...]

# 对特征重要性进行倒序排序
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# 提取特征名称和重要性
features = [pair[0] for pair in feature_importances]
importances = [pair[1] for pair in feature_importances]

# 设置颜色映射，这里使用 'rainbow' 映射，颜色更加艳丽
colors = plt.cm.rainbow(np.linspace(0, 1, len(features)))

# 设置条形图的宽度
bar_width = 0.8

# 绘制横向条形图
plt.figure(figsize=(10, 12))  # 可以调整图形大小
plt.barh(range(len(features))[::-1], importances, color=colors, height=bar_width, align='center')

# 添加标题和标签
plt.title('Feature Importances (Descending)')
plt.xlabel('Importance')
plt.yticks(range(len(features))[::-1], [f for f, _ in feature_importances])  # 倒序显示特征名称

# 设置条形图之间的间距
plt.subplots_adjust(left=0.35)  # 增加左侧空白区域

# 显示图形
plt.tight_layout()
plt.show()