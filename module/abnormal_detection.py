# Author:Jane
# -*- coding = utf-8 -*-
# @Time :2024/3/5 18:43
# @Author :langyanping1
# @Site :
# @File :abnormal_detection.py
# @software :



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


from module.undersample_value import under_sample_split

from module.best_pridict_model import abs_predict

warnings.filterwarnings('ignore')

################   1- 下采样数据  +++++++++++++++++++++++++
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = under_sample_split()


########################   异常检测部分    ######################################

abs_difference = abs_predict()

######################   构建模型  +++++++++++++++++++++++++++++++

def fit_predict(abs_difference):
    ############## N-Sigma方法
    # N-Sigma 方法通常用于基于数据的正态分布进行异常检测。在这种方法中，我们计算数据的均值和标准差，然后使用一定的倍数（通常是3或者6倍）标准差来定义正常值的范围，超出这个范围的数据被认为是异常值。
    # 假设 abs_difference 是你的绝对差异数据
    abs_difference = abs_difference.values.reshape(-1, 1)
    mean = np.mean(abs_difference)
    std = np.std(abs_difference)

    # 定义 N 倍标准差
    n_sigma = 3
    threshold = mean + n_sigma * std

    # 标记异常值
    outliers_ns = (abs_difference > threshold)

    ############## One-Class SVM
    # One-Class SVM 是一种无监督学习算法，它试图学习数据的特征，并将其特征空间划分为正常和异常区域。



    # 创建 One-Class SVM 模型并拟合数据
    model_one_class_svm = OneClassSVM(nu=0.01)  # nu 是一个超参数，用于控制异常点的比例
    model_one_class_svm.fit(abs_difference.reshape(-1, 1))

    # 预测数据点的标签（1表示正常，-1表示异常）
    outliers_one_class_svm = model_one_class_svm.predict(abs_difference.reshape(-1, 1))

    ############## Isolation Forest
    # 隔离森林是一种基于树的集成算法，它通过将数据逐渐分割为单独的区域来识别异常点。



    # 创建 Isolation Forest 模型并拟合数据
    model_isolation_forest = IsolationForest(contamination=0.01)  # contamination 用于控制异常点的比例
    model_isolation_forest.fit(abs_difference.reshape(-1, 1))

    # 预测数据点的标签（1表示正常，-1表示异常）
    outliers_isolation_forest = model_isolation_forest.predict(abs_difference.reshape(-1, 1))

    ############## LOF (局部异常因子)


    # 创建 LOF 模型并拟合数据
    model_lof = LocalOutlierFactor(contamination=0.01)  # contamination 控制异常点的比例
    outliers_lof = model_lof.fit_predict(abs_difference.reshape(-1, 1))

fit_predict(abs_difference)


def fit_predict2(abs_difference):
    # 假设 abs_difference 是您的绝对差异数据的时间序列
    # 将 abs_difference 转换为 pandas 的 DataFrame，并设置时间索引
    abs_difference = pd.Series(abs_difference)

    # N-Sigma 方法
    mean = abs_difference.mean()
    std = abs_difference.std()

    # 根据均值和标准差定义异常阈值
    n_sigma = 3
    threshold = mean + n_sigma * std

    # 标记异常值
    outliers_ns = abs_difference[abs_difference > threshold]

    # One-Class SVM
    model_one_class_svm = OneClassSVM(nu=0.01)  # nu 是一个超参数，用于控制异常点的比例
    outliers_one_class_svm = model_one_class_svm.fit_predict(abs_difference.values.reshape(-1, 1))
    # 统计异常值数量
    num_outliers_one_class_svm = sum(outliers_one_class_svm == -1)

    # Isolation Forest
    model_isolation_forest = IsolationForest(contamination=0.01)  # contamination 用于控制异常点的比例
    outliers_isolation_forest = model_isolation_forest.fit_predict(abs_difference.values.reshape(-1, 1))
    # 统计异常值数量
    num_outliers_isolation_forest = sum(outliers_isolation_forest == -1)

    # LOF
    model_lof = LocalOutlierFactor(contamination=0.01)  # contamination 控制异常点的比例
    outliers_lof = model_lof.fit_predict(abs_difference.values.reshape(-1, 1))
    # 统计异常值数量
    num_outliers_lof = sum(outliers_lof == -1)


fit_predict2(abs_difference)


sys.exit()



############## 绘图观察上述四种模型 ################
# 绘制绝对差异数据的直方图
plt.figure(figsize=(10, 6))
plt.hist(abs_difference, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Absolute Differences')
plt.xlabel('Absolute Difference')
plt.ylabel('Frequency')
plt.show()

# 绘制异常检测结果
plt.figure(figsize=(10, 6))

# N-Sigma 方法
plt.subplot(2, 2, 1)
plt.scatter(range(len(abs_difference)), abs_difference, c=outliers_ns, cmap='coolwarm')
plt.title('N-Sigma Method')
plt.xlabel('Index')
plt.ylabel('Absolute Difference')

# One-Class SVM
plt.subplot(2, 2, 2)
plt.scatter(range(len(abs_difference)), abs_difference, c=outliers_one_class_svm, cmap='coolwarm')
plt.title('One-Class SVM')
plt.xlabel('Index')
plt.ylabel('Absolute Difference')

# Isolation Forest
plt.subplot(2, 2, 3)
plt.scatter(range(len(abs_difference)), abs_difference, c=outliers_isolation_forest, cmap='coolwarm')
plt.title('Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Absolute Difference')

# LOF
plt.subplot(2, 2, 4)
plt.scatter(range(len(abs_difference)), abs_difference, c=outliers_lof, cmap='coolwarm')
plt.title('Local Outlier Factor (LOF)')
plt.xlabel('Index')
plt.ylabel('Absolute Difference')

plt.tight_layout()
plt.show()

########################   使用GridSearchCV计算最优参数    ######################################
# 下述代码中的 GridSearchCV 就是使用了交叉验证来寻找最佳参数和最佳评分。在 GridSearchCV 中，参数 cv=5 指定了使用 5 折交叉验证。这意味着数据集会被分成 5 份，然后模型会被训练 5 次，每次都使用其中 4 份作为训练集，1 份作为验证集。
# 因此，GridSearchCV 使用了交叉验证来评估每个参数组合的性能，并找到最佳参数和相应的评分。在代码中，我们使用了 scoring='accuracy' 来评估模型性能，但你可以根据实际情况选择适合你问题的评分指标。

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error

# 1. N-Sigma模型的参数选择
# 对于N-Sigma模型，参数通常是事先设定的，例如标准差的倍数等

# 2. One-Class SVM的参数选择
model_svm = OneClassSVM()
param_grid_svm = {
    'nu': [0.01, 0.1, 0.2, 0.3, 0.5],
    'kernel': ['rbf', 'sigmoid', 'linear']
}
grid_search_svm = GridSearchCV(model_svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_test_undersample, y_test_undersample)

# 3. Isolation Forest的参数选择
model_forest = IsolationForest()
param_grid_forest = {
    'n_estimators': [100, 200, 300],
    'max_samples': [100, 200, 300],
    'contamination': [0.01, 0.05, 0.1]
}


# 自定义评分函数
def custom_scorer(y_true, y_pred):
    # 假设 y_true 是二进制的（异常为1，正常为0）
    return roc_auc_score(y_true, y_pred)


# 创建评分器
scorer = make_scorer(custom_scorer, needs_threshold=True)

grid_search_forest = GridSearchCV(model_forest, param_grid_forest, cv=5, scoring=scorer)
# grid_search_forest = GridSearchCV(model_forest, param_grid_forest, cv=5, scoring='accuracy')
grid_search_forest.fit(X_test_undersample, y_test_undersample)

# 4. LOF模型的参数选择
model_lof = LocalOutlierFactor()
param_grid_lof = {
    'n_neighbors': [5, 10, 20],
    'contamination': [0.01, 0.05, 0.1]
}
grid_search_lof = GridSearchCV(model_lof, param_grid_lof, cv=5, scoring='accuracy')
grid_search_lof.fit(X_test_undersample, y_test_undersample)

# 最佳参数
best_params_svm = grid_search_svm.best_params_
best_params_forest = grid_search_forest.best_params_
best_params_lof = grid_search_lof.best_params_

# 最佳评分
best_score_svm = grid_search_svm.best_score_
best_score_forest = grid_search_forest.best_score_
best_score_lof = grid_search_lof.best_score_

# 输出每个模型的最佳参数和最佳评分
print("Best params for One-Class SVM:", best_params_svm)
print("Best score for One-Class SVM:", best_score_svm)

print("Best params for Isolation Forest:", best_params_forest)
print("Best score for Isolation Forest:", best_score_forest)

print("Best params for LOF:", best_params_lof)
print("Best score for LOF:", best_score_lof)

########################   最优参数绘图    ######################################

models = ['One-Class SVM', 'Isolation Forest', 'LOF']

nu_values = [best_params_svm['nu'], 0, 0]  # 假设nu是One-Class SVM的关键参数
n_estimators_values = [0, best_params_forest['n_estimators'], 0]  # 假设n_estimators是Isolation Forest的关键参数
n_neighbors_values = [0, 0, best_params_lof['n_neighbors']]  # 假设n_neighbors是LOF的关键参数

barWidth = 0.25

r1 = np.arange(len(nu_values))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, nu_values, color='b', width=barWidth, edgecolor='grey', label='One-Class SVM')
plt.bar(r2, n_estimators_values, color='r', width=barWidth, edgecolor='grey', label='Isolation Forest')
plt.bar(r3, n_neighbors_values, color='g', width=barWidth, edgecolor='grey', label='LOF')

plt.xlabel('Models', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(nu_values))], models)

plt.legend()
plt.show()

#### 绘图
# 如何判断使用哪一个
# 如果你想知道模型在不同数量的训练数据下的表现，以判断模型是否需要更多的数据或者是否有过拟合问题，你应该使用 learning_curve。
# 如果你想知道模型在某个特定参数的不同值下的表现，以找到最佳参数设置，你应该使用 validation_curve。
# 通常，learning_curve 用于数据量的问题，而 validation_curve 用于模型参数调优。两者都是模型评估和选择过程中的重要工具。

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


def plot_learning_curve(model, X, y, scorer):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5,
                                                            scoring=scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def plot_validation_curve(model, X, y, param_name, param_range, scorer):
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, scoring=scorer, cv=5
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


# 使用示例
# 假设你已经定义了最佳参数的模型 best_model，以及数据 X 和 y
# 现暂时以 best_model = grid_search_forest.best_estimator_ 进行模拟

# 使用自定义评分器调用函数
scorer = make_scorer(custom_scorer, needs_threshold=True)

best_model = grid_search_forest.best_estimator_
plot_learning_curve(best_model, X_test_undersample, y_test_undersample, scorer)
plot_validation_curve(best_model, X_test_undersample, y_test_undersample, 'n_estimators', [100, 200, 300, 400, 500],
                      scorer)
