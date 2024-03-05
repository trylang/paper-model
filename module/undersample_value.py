import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

data = pd.read_csv("../data/creditcard_327.csv")
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
def X_Y_undersample():
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

    print(11, np.shape(X_undersample), type(X_undersample))
    print(22, np.shape(y_undersample), type(y_undersample))

    # 下采样 样本比例
    print("正常样本所占整体比例: ",
          len(under_sample_data[(under_sample_data.V2 > -1) & (under_sample_data.V2 < 1)]) / len(under_sample_data))
    print("异常样本所占整体比例: ",
          len(under_sample_data[(under_sample_data.V2 < -1) | (under_sample_data.V2 > 1)]) / len(under_sample_data))
    print("下采样策略总体样本数量: ", len(under_sample_data))

    return X, y, X_undersample, y_undersample, under_sample_data




# 数据集划分
def primary_data_split():
    X, y, X_undersample, y_undersample, under_sample_data = X_Y_undersample()

    # 整个数据集进行划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # print("原始训练集包含样本数量: ", len(X_train))
    # print("原始测试集包含样本数量: ", len(X_test))
    # print("原始样本总数: ", len(X_train)+len(X_test))
    return X_train, X_test, y_train, y_test

def under_sample_split():
    X, y, X_undersample, y_undersample, under_sample_data = X_Y_undersample()

    # 下采样数据集进行划分
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                        , y_undersample
                                                                                                        , test_size=0.3
                                                                                                        ,random_state=0)
    # print("")
    # print("下采样训练集包含样本数量: ", len(X_train_undersample))
    # print("下采样测试集包含样本数量: ", len(X_test_undersample))
    # print("下采样样本总数: ", len(X_train_undersample)+len(X_test_undersample))
    return X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample