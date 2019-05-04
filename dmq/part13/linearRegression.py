# -*- coding: utf-8 -*-
# @Author: wangchao
# @Time: 19-5-3 下午9:18

# 导入库
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np

# 导入数据集
data = pd.read_csv('pollution.csv')
# 去除与训练无关的因素
# dataset = data.drop(["No", "year", "month", "day", "hour", "cbwd"], axis=1)
dataset = data.drop(["No", "cbwd"], axis=1)
# 将pm2.5移到第一列
pm = dataset.pop('pm2.5')
dataset.insert(0, 'pm2.5', pm)

# 补填缺失值
dataset = dataset.fillna(0)

X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0:1].values
# 分割训练集和测试集(2014年全为测试集)
X_train, X_test, Y_train, Y_test = X[:35065], X[35065:], Y[:35065], Y[35065:]

# 选取100个数据做测试(数据量太大图看不清)
X_test = X_test[:100]
Y_test = Y_test[:100]
# 训练模型
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
# 计算mse
x = mse(Y_test, Y_pred)
print(x)
# 画图
epochs = range(1, len(Y_pred) + 1)
plt.plot(epochs, Y_test, "bo", label='test data')
plt.plot(epochs, Y_pred, 'r', label='pred data')
plt.title('test and pred loss')
plt.xlabel('Epochs')
plt.ylabel('data')
plt.legend()
plt.show()
