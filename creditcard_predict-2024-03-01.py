# Author:Jane
# -*- coding = utf-8 -*-
# @Time :2024/3/1 21:44
# @Author :langyanping1
# @Site :
# @File :creditcard_predict.py
# @software :
# 修改creditcard_predict.py 文件中 将预测指标Class 换为V2
# 完整全套代码

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

warnings.filterwarnings('ignore')

data = pd.read_csv("./data/creditcard_327.csv")
data.head()

# 数据标签分布
count_classes = pd.Series(data['V2']).value_counts().sort_index()
print(count_classes)

# 论文叙述1： 异常样本相对正常样本很少 284315：492
count_classes.plot(kind='bar')
plt.title("Fraud class histogram")
plt.xlabel("V2")
plt.ylabel("Frequency")
# plt.show()


# 论文叙述2： 数据标准化处理
from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
print(data.head())


# 下采样方案
def X_Y_undersample(data):
    X = data.iloc[:, data.columns != 'V2']
    y = data.iloc[:, data.columns == 'V2']

    # 得到所有异常样本的索引
    number_records_fraud = len(data[(data.V2 > 1) | (data.V2 < -1)])
    fraud_indices = np.array(data[(data.V2 > 1) | (data.V2 < -1)].index)

    # 得到所有正常样本的索引
    normal_indices = data[(data.V2 > -1) & (data.V2 < 1)].index

    # 在正常样本中随机采样出指定个数的样本，并取其索引
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)

    # 有了正常和异常样本后把它们的索引都拿到手
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

    # 根据索引得到下采样所有样本点
    under_sample_data = data.iloc[under_sample_indices, :]

    X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'V2']
    y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'V2']
    return X, y, X_undersample, y_undersample, under_sample_data


X, y, X_undersample, y_undersample, under_sample_data = X_Y_undersample(data)
# 下采样 样本比例
print("正常样本所占整体比例: ",
      len(under_sample_data[(under_sample_data.V2 > -1) & (under_sample_data.V2 < 1)]) / len(under_sample_data))
print("异常样本所占整体比例: ",
      len(under_sample_data[(under_sample_data.V2 < -1) | (under_sample_data.V2 > 1)]) / len(under_sample_data))
print("下采样策略总体样本数量: ", len(under_sample_data))

# 数据集划分
from sklearn.model_selection import train_test_split

# 整个数据集进行划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print("原始训练集包含样本数量: ", len(X_train))
# print("原始测试集包含样本数量: ", len(X_test))
# print("原始样本总数: ", len(X_train)+len(X_test))

# 下采样数据集进行划分
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                    , y_undersample
                                                                                                    , test_size=0.3
                                                                                                    , random_state=0)
# print("")
# print("下采样训练集包含样本数量: ", len(X_train_undersample))
# print("下采样测试集包含样本数量: ", len(X_test_undersample))
# print("下采样样本总数: ", len(X_train_undersample)+len(X_test_undersample))


########################   预测模型最佳选取    ######################################

# 论文叙述3： 5折交叉验证
# 通过５折交叉验证，实现逻辑回归，决策树，SVM,随机森林，GBDT,Xgboost,lightGBM的评分
# https://blog.csdn.net/weixin_41710583/article/details/85016622

# cross_val_score的 scoring参数值解析
# https://blog.csdn.net/qq_32590631/article/details/82831613

import pandas as pd
import warnings
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# ARIMA    pip install statsmodels
from statsmodels.tsa.arima.model import ARIMA

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

# 导入算法
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit, cross_validate

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.base import BaseEstimator, RegressorMixin
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import statsmodels.api as sm
from lightgbm import LGBMRegressor

# 设置随机种子
SEED = 2018

df_X = data.iloc[:, data.columns != 'V2']
df_y = data.iloc[:, data.columns == 'V2']

df_X = scale(df_X, axis=0)  # 将数据转化为标准数据

# 构建模型
lr = LogisticRegression(random_state=SEED, tol=1e-6)  # 逻辑回归模型

# # 创建ARIMA模型
# class ARIMAWrap(BaseEstimator, RegressorMixin):
#     def __init__(self, order=(5, 1, 0)):
#         self.order = order
#
#     def fit(self, X, y):
#         self.model = ARIMA(y, order=self.order)
#         self.fitted_model = self.model.fit()
#         return self
#
#     def predict(self, X):
#         # 在这里，你可以根据需要指定如何进行预测
#         # 这里假设 X 是测试数据，实际情况根据 ARIMA 模型的预测方法进行调整
#         predictions = self.fitted_model.forecast(steps=len(X))
#         return predictions
#
# # 创建 ARIMAWrap 实例
# arima = ARIMAWrap(order=(5, 1, 0))



svm = SVC(probability=True, random_state=SEED, tol=1e-6)  # SVM模型

forest = RandomForestRegressor(n_estimators=100, random_state=SEED)  # 随机森林

Gbdt = GradientBoostingClassifier(random_state=SEED)  # CBDT

Xgbc = XGBClassifier(random_state=SEED)  # Xgbc

gbm = lgb.LGBMClassifier(random_state=SEED)  # lgb


# 定义计算 RMSE 的自定义评估函数
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# 使用 cross_val_predict 进行交叉验证预测
# predicted_arima = cross_val_predict(arima, X_train_undersample, y_train_undersample, cv=5)

# # 使用自定义评估函数计算 RMSE
# rmse_score_arima = rmse(y_train_undersample, predicted_arima)
# rmse_score_lstm = rmse(y_train_undersample, predicted_lstm)


# model_names = ["arima", "lstm"]
# model_name = ["lr", "arima", "lstm", "svm", "forest", "Gbdt", "Xgbc", "gbm"]

# 创建一个评分函数，使用均方根误差(RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 创建TimeSeriesSplit交叉验证生成器
tscv = TimeSeriesSplit(n_splits=5)

# 设置LSTM模型参数
lstm = Sequential()
lstm.add(LSTM(units=100, input_shape=(X_test_undersample.shape[1], X_test_undersample.shape[2])))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

# 设置ARIMA模型参数
arima = ARIMA(y_test_undersample, order=(5,1,0))

# 设置LightGBM模型参数
lgbm = LGBMRegressor(num_leaves=31, learning_rate=0.1, n_estimators=100)



# 使用 cross_val_predict 进行交叉验证预测
lstm_y_pred = cross_val_predict(estimator=lstm, X=X_test_undersample, y=y_test_undersample, cv=tscv)
arima_y_pred = cross_val_predict(estimator=arima, X=X_test_undersample, y=y_test_undersample, cv=tscv)
lgbm_y_pred = cross_val_predict(estimator=lgbm, X=X_test_undersample, y=y_test_undersample, cv=tscv)

# 使用自定义的评分函数计算评分
lstm_score = rmse(y_test_undersample, lstm_y_pred)
arima_score = rmse(y_test_undersample, arima_y_pred)
lgbm_score = rmse(y_test_undersample, lgbm_y_pred)

# 打印评分
print('RMSE for LSTM:', lstm_score)
print('RMSE for ARIMA:', arima_score)
print('RMSE for LightGBM:', lgbm_score)

# 退出程序，状态码为0（表示正常退出）
sys.exit()


# 论文叙述4： 混淆矩阵(可出图)  -- 在上面得出最优预测算法之后，进行一下混淆矩阵进行证明。然后做后面的异常检测
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


import itertools


# 下采样方案在测试集中的结果
def plot_undertestsample_result():
    lr = XGBClassifier(random_state=SEED)
    lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    y_pred_undersample = lr.predict(X_test_undersample.values)

    # 计算所需值
    cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
    np.set_printoptions(precision=2)

    print("召回率1: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # 绘制
    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()


plot_undertestsample_result()
#
#
# # 下采样方案在原始数据集中的结果
# def plot_initial_result(best_c):
#     lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
#     lr.fit(X_train_undersample, y_train_undersample.values.ravel())
#     y_pred = lr.predict(X_test.values)
#
#     # 计算所需值
#     cnf_matrix = confusion_matrix(y_test, y_pred)
#     np.set_printoptions(precision=2)
#
#     print("召回率2: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
#
#     # 绘制
#     class_names = [0, 1]
#     plt.figure()
#     plot_confusion_matrix(cnf_matrix
#                           , classes=class_names
#                           , title='Confusion matrix')
#     plt.show()
#
#
# # plot_initial_result(best_c)
#
#


########################   计算绝对差值    ######################################

### 使用下采样数据，使用 随机森林进行预测并计入表格中展示

forestValue = RandomForestRegressor(n_estimators=100, random_state=0)

forestValue.fit(X_train_undersample, y_train_undersample)
y_pred = forestValue.predict(X_test_undersample)

X_test_undersample['pred_label'] = y_pred

# 打印出形状以帮助调试
print('y_pred shape:', np.shape(y_pred))
print('X_test_undersample shape:', X_test_undersample.shape)
print('y_test_undersample shape:', np.shape(y_test_undersample))

# y_pred shape: (296,)
# X_test_undersample shape: (296, 30)
# y_test_undersample shape: (296, 1)


print(X_test_undersample.head())

# print(len(X_test_undersample))
# print(len(y_pred))
# print(len(y_test_undersample))


# 将单列 DataFrame 转换为 Series
y_test_undersample_series = y_test_undersample.iloc[:, 0]

# 现在 y_pred 和 y_test_undersample_flattened 都是一维数组，可以相减
## 再计算V2 绝对差值
abs_difference = np.abs(y_pred - y_test_undersample_series)
print(55, abs_difference)

########################   异常检测部分    ######################################

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

from sklearn.svm import OneClassSVM

# 创建 One-Class SVM 模型并拟合数据
model_one_class_svm = OneClassSVM(nu=0.01)  # nu 是一个超参数，用于控制异常点的比例
model_one_class_svm.fit(abs_difference.reshape(-1, 1))

# 预测数据点的标签（1表示正常，-1表示异常）
outliers_one_class_svm = model_one_class_svm.predict(abs_difference.reshape(-1, 1))

############## Isolation Forest
# 隔离森林是一种基于树的集成算法，它通过将数据逐渐分割为单独的区域来识别异常点。

from sklearn.ensemble import IsolationForest

# 创建 Isolation Forest 模型并拟合数据
model_isolation_forest = IsolationForest(contamination=0.01)  # contamination 用于控制异常点的比例
model_isolation_forest.fit(abs_difference.reshape(-1, 1))

# 预测数据点的标签（1表示正常，-1表示异常）
outliers_isolation_forest = model_isolation_forest.predict(abs_difference.reshape(-1, 1))

############## LOF (局部异常因子)
from sklearn.neighbors import LocalOutlierFactor

# 创建 LOF 模型并拟合数据
model_lof = LocalOutlierFactor(contamination=0.01)  # contamination 控制异常点的比例
outliers_lof = model_lof.fit_predict(abs_difference.reshape(-1, 1))

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
