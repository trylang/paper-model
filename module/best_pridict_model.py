# Author:Jane
# -*- coding = utf-8 -*-
# @Time :2024/3/4 15:24
# @Author :langyanping1
# @Site :
# @File :creditcard_predict.py
# @software :
# 完成LSTM最佳8个参数，并计算预测值和差值绝对值

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

from module.undersample_value import under_sample_split

warnings.filterwarnings('ignore')

################   1- 下采样数据  +++++++++++++++++++++++++
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = under_sample_split()



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
from tensorflow.keras.regularizers import L1L2

# 设置随机种子
SEED = 2018

# 构建模型
lr = LogisticRegression(random_state=SEED, tol=1e-6)  # 逻辑回归模型


##################   创建预测模型  ++++++++++++++++++

# 设置 K-Fold 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 初始化评分记录器
scores = {'LSTM': [], 'ARIMA': [], 'LightGBM': []}

# LSTM和ARIMA模型通常要求序列数据作为输入，所以确保这些前提条件
for train_index, test_index in tscv.split(X_test_undersample):
    # 分割数据集
    X_train, X_test = X_test_undersample.iloc[train_index], X_test_undersample.iloc[test_index]
    y_train, y_test = y_test_undersample.iloc[train_index], y_test_undersample.iloc[test_index]

    # LSTM模型
    # 重造数据以符合LSTM的输入需求: [samples, time steps, features]
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train_lstm, y_train, epochs=20, verbose=0)
    y_pred_lstm = lstm_model.predict(X_test_lstm)

    scores['LSTM'].append({'MAPE': mean_absolute_percentage_error(y_test, y_pred_lstm),
                           'MSE': mean_squared_error(y_test, y_pred_lstm)})

    # ARIMA模型（请根据您的具体数据调整模型参数）
    arima_model = ARIMA(y_train, order=(5, 1, 0))  # 这里的(5,1,0)是(p,d,q)参数的示例
    arima_model_fit = arima_model.fit()
    y_pred_arima = arima_model_fit.forecast(steps=len(y_test))

    scores['ARIMA'].append({'MAPE': mean_absolute_percentage_error(y_test, y_pred_arima),
                            'MSE': mean_squared_error(y_test, y_pred_arima)})

    # LightGBM模型
    lgbm_model = LGBMRegressor(n_estimators=100)
    lgbm_model.fit(X_train, y_train)
    y_pred_lgbm = lgbm_model.predict(X_test)

    scores['LightGBM'].append({'MAPE': mean_absolute_percentage_error(y_test, y_pred_lgbm),
                               'MSE': mean_squared_error(y_test, y_pred_lgbm)})


for model_name, score_list in scores.items():
    mape_scores = [score['MAPE'] for score in score_list]
    mse_scores = [score['MSE'] for score in score_list]
    print(f"{model_name} MAPE: {np.mean(mape_scores):.4f}")
    print(f"{model_name} MSE: {np.mean(mse_scores):.4f}")

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

# 准备数据
X_train = X_test_undersample.values  # assuming X_test_undersample is a DataFrame
y_train = y_test_undersample.values  # assuming y_test_undersample is a DataFrame

X_test_3D = X_test_undersample.values.reshape(X_test_undersample.shape[0], 1, X_test_undersample.shape[1])
y_test_values = y_test_undersample.values  # 将DataFrame转换为numpy数组

# 定义创建 LSTM 模型的函数
def create_lstm_model(units=50, dropout=0.0, recurrent_dropout=0.0, optimizer='adam', activation='tanh', recurrent_activation='sigmoid'
                      # return_sequences=False,
                      # stateful=False
                      , recurrent_regularizer=None, kernel_regularizer=None, bias_regularizer=None):

    model = Sequential()
    model.add(LSTM(units,
                   dropout=dropout,
                   recurrent_dropout=recurrent_dropout,
                   activation=activation,
                   recurrent_activation=recurrent_activation,
                   # return_sequences=return_sequences,
                   # stateful=stateful,
                   recurrent_regularizer=recurrent_regularizer,
                   kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer,
                   input_shape=(X_test_3D.shape[1], X_test_3D.shape[2])))
    model.add(Dense(1))
    if optimizer == 'adam':
        opt = Adam()
    else:
        opt = RMSprop()

    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

# 定义评分函数
def custom_scorer(y_true, y_pred):
    return -mean_squared_error(y_true, y_pred)  # negated mean squared error for optimization

# 定义参数空间
# 这里：return_sequences 和 stateful 放开参数会报错，顾在此只使用这8个参数
param_grid = {
    'units': [50, 100, 150],  # LSTM 单元数量
    'dropout': [0.0, 0.1, 0.2],  # dropout 比例
    'recurrent_dropout': [0.0, 0.1, 0.2],  # recurrent_dropout 比例
    'activation': ['relu', 'tanh'],  # 激活函数类型
    'recurrent_activation': ['sigmoid', 'tanh'],  # recurrent 激活函数类型
    # 'return_sequences': [True, False],  # 是否返回全部序列
    # 'stateful': [True, False],  # 是否为 stateful LSTM
    # 其他正则化参数根据需要增加，这里简化为 None
    'recurrent_regularizer': [None, L1L2(l1=0.01, l2=0.01)],
    'kernel_regularizer': [None, L1L2(l1=0.01, l2=0.01)],
    'bias_regularizer': [None, L1L2(l1=0.01, l2=0.01)]
}

# 设置初始的最佳分数和最佳参数
best_score = float('inf')
best_params = {}

param_list = list(ParameterSampler(param_grid, n_iter=10, random_state=42))  # 从参数网格中随机抽样 10 个参数组合

for params in param_list:

    units = params['units']
    dropout = params['dropout']
    recurrent_dropout = params['recurrent_dropout']
    activation = params['activation']

    recurrent_activation = params['recurrent_activation']
    # return_sequences = params['return_sequences']
    # stateful = params['stateful']
    recurrent_regularizer = params['recurrent_regularizer']

    kernel_regularizer = params['kernel_regularizer']
    bias_regularizer = params['bias_regularizer']


    model = create_lstm_model(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout,
                              activation=activation, recurrent_activation=recurrent_activation,
                      # return_sequences=return_sequences
    # stateful=stateful
                            recurrent_regularizer=recurrent_regularizer,
                      kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

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

# Best parameters:  {'units': 150, 'recurrent_regularizer': <keras.src.regularizers.L1L2 object at 0x156226dc0>, 'recurrent_dropout': 0.2, 'recurrent_activation': 'sigmoid', 'kernel_regularizer': None, 'dropout': 0.2, 'bias_regularizer': None, 'activation': 'tanh'}
# Best score:  1.2837531333222396

########################   预测及绝对值   +++++++++++++++++++++

# 使用最佳参数构建 LSTM 模型
best_units = 150  # 假设最佳单位数为150
best_optimizer = 'adam'  # 假设最佳优化器为adam
X_test_3D = X_test_undersample.values.reshape(X_test_undersample.shape[0], 1, X_test_undersample.shape[1])
y_test_values = y_test_undersample.values  # 将DataFrame转换为numpy数组

def abs_predict():
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

    # 将单列 DataFrame 转换为 Series
    y_test_undersample_series = y_test_undersample.iloc[:, 0]

    ## 再计算V2 绝对差值
    abs_difference = np.abs(predictions - y_test_undersample)
    print(55, abs_difference)

    # 输出预测值和差值
    print("Predictions: ", predictions)
    print("Absolute Errors: ", abs_difference)

    return abs_difference

