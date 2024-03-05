# Author:Jane
# -*- coding = utf-8 -*-
# @Time :2024/3/4 15:24
# @Author :langyanping1
# @Site :
# @File :creditcard_predict.py
# @software :
# 完成LSTM最佳2个参数，并计算预测值和差值绝对值

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
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit, cross_validate, GridSearchCV, RandomizedSearchCV

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from scipy.stats import randint as sp_randint
from sklearn.model_selection import ParameterSampler

# 设置随机种子
SEED = 2018

df_X = data.iloc[:, data.columns != 'V2']
df_y = data.iloc[:, data.columns == 'V2']

df_X = scale(df_X, axis=0)  # 将数据转化为标准数据

# 构建模型
lr = LogisticRegression(random_state=SEED, tol=1e-6)  # 逻辑回归模型


##################   创建预测模型  ++++++++++++++++++

print(11, np.shape(X_test_undersample), type(X_test_undersample))
print(22, np.shape(y_test_undersample), type(y_test_undersample))


# 设置 K-Fold 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 初始化评分记录器
scores = {'LSTM': [], 'ARIMA': [], 'LightGBM': []}

# # LSTM和ARIMA模型通常要求序列数据作为输入，所以确保这些前提条件
# for train_index, test_index in tscv.split(X_test_undersample):
#     # 分割数据集
#     X_train, X_test = X_test_undersample.iloc[train_index], X_test_undersample.iloc[test_index]
#     y_train, y_test = y_test_undersample.iloc[train_index], y_test_undersample.iloc[test_index]
#
#     # LSTM模型
#     # 重造数据以符合LSTM的输入需求: [samples, time steps, features]
#     X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
#     X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
#
#     lstm_model = Sequential([
#         LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
#         Dense(1)
#     ])
#     lstm_model.compile(optimizer='adam', loss='mean_squared_error')
#     lstm_model.fit(X_train_lstm, y_train, epochs=20, verbose=0)
#     y_pred_lstm = lstm_model.predict(X_test_lstm)
#
#     scores['LSTM'].append({'MAPE': mean_absolute_percentage_error(y_test, y_pred_lstm),
#                            'MSE': mean_squared_error(y_test, y_pred_lstm)})
#
#     # ARIMA模型（请根据您的具体数据调整模型参数）
#     arima_model = ARIMA(y_train, order=(5, 1, 0))  # 这里的(5,1,0)是(p,d,q)参数的示例
#     arima_model_fit = arima_model.fit()
#     y_pred_arima = arima_model_fit.forecast(steps=len(y_test))
#
#     scores['ARIMA'].append({'MAPE': mean_absolute_percentage_error(y_test, y_pred_arima),
#                             'MSE': mean_squared_error(y_test, y_pred_arima)})
#
#     # LightGBM模型
#     lgbm_model = LGBMRegressor(n_estimators=100)
#     lgbm_model.fit(X_train, y_train)
#     y_pred_lgbm = lgbm_model.predict(X_test)
#
#     scores['LightGBM'].append({'MAPE': mean_absolute_percentage_error(y_test, y_pred_lgbm),
#                                'MSE': mean_squared_error(y_test, y_pred_lgbm)})
#
#
# for model_name, score_list in scores.items():
#     mape_scores = [score['MAPE'] for score in score_list]
#     mse_scores = [score['MSE'] for score in score_list]
#     print(f"{model_name} MAPE: {np.mean(mape_scores):.4f}")
#     print(f"{model_name} MSE: {np.mean(mse_scores):.4f}")

#   选取LSTM进行预测
# LSTM MAPE: 0.9669
# LSTM MSE: 2.2171
# ARIMA MAPE: 1.6417
# ARIMA MSE: 2.6676
# LightGBM MAPE: 1.4516
# LightGBM MSE: 2.4651



#################  使用LSTM 预测 ，并计算绝对值+++++++++++++++
# 数据准备
# 假设X_test_undersample和y_test_undersample已经是加载好的pandas DataFrame

# 数据预处理
scaler = MinMaxScaler()
X_test_scaled = scaler.fit_transform(X_test_undersample)
X_test_3D = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
y_test_values = y_test_undersample.values  # 将 DataFrame 转换为 numpy 数组


# 创建一个LSTM模型函数
def create_lstm_model(units=50, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, input_shape=(X_test_3D.shape[1], X_test_3D.shape[2])))
    model.add(Dense(1))

    if optimizer == 'adam':
        opt = Adam()
    else:
        opt = RMSprop()

    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


# 设置参数网格
param_grid = {
    "units": [50, 100, 150],  # LSTM units
    "optimizer": ['adam', 'rmsprop']  # 优化器选择
}

# 设置初始的最佳分数和最佳参数
best_score = float('inf')
best_params = {}

param_list = list(ParameterSampler(param_grid, n_iter=10, random_state=42))  # 从参数网格中随机抽样 10 个参数组合

for params in param_list:
    units = params['units']
    optimizer = params['optimizer']

    model = create_lstm_model(units=units, optimizer=optimizer)
    model.fit(X_test_3D, y_test_values, epochs=10, batch_size=32, verbose=0)

    # 计算模型预测值并计算均方误差
    predictions = model.predict(X_test_3D)
    mse = mean_squared_error(y_test_values, predictions)

    # 更新 best_score 和 best_params
    if mse < best_score:
        best_score = mse
        best_params = params

# 输出最佳结果
print("Best parameters: ", best_params)
print("Best score: ", best_score)

#################### 使用最佳参数构建 LSTM 模型 预测，并计算绝对值 +++++++++++++++++++++++
best_units = 150  # 假设最佳单位数为150
best_optimizer = 'adam'  # 假设最佳优化器为adam

model = Sequential()
model.add(LSTM(best_units, input_shape=(X_test_3D.shape[1], X_test_3D.shape[2])))
model.add(Dense(1))

if best_optimizer == 'adam':
    opt = Adam()
else:
    opt = RMSprop()

model.compile(loss='mean_squared_error', optimizer=opt)

# 在整个数据集上训练模型
model.fit(X_test_3D, y_test_values, epochs=10, batch_size=32, verbose=0)

# 使用模型进行预测
predictions = model.predict(X_test_3D)

# 计算预测值和真实值之间的差值的绝对值
absolute_errors = np.abs(predictions.flatten() - y_test_values)

# 输出预测值和差值
print("Predictions: ", predictions)
print("Absolute Errors: ", absolute_errors)