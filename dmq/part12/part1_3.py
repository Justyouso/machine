# -*- coding: utf-8 -*-
# @Time: 19-4-28 下午10:08
from sklearn.cluster import KMeans
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# # 加载数据
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# # 选取数据
# train_images = train_images[:5000]
# train_labels = train_labels[:5000]
# test_images = test_images[:2000]
# test_labels = test_labels[:2000]
#
# # 准备图像数据(图像28*28,取值区间[0-1])
# train_images = train_images.reshape((5000, 28 * 28))
# train_images = train_images.astype("float32") / 255
#
# test_images = test_images.reshape((2000, 28 * 28))
# test_images = test_images.astype("float32") / 255
#
# # 准备标签
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
#
# x = np.concatenate((train_images, test_images))
# y = np.concatenate((train_labels, test_labels))
#
# # x = x.reshape((x.shape[0], -1))
# # x = np.divide(x, 255.)
#
# n_clusters = len(np.unique(y))
#
# kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
#
# y_pred = kmeans.fit_predict(x)
#
# acc = accuracy_score(y, y_pred)
#
# # 获取真正率和假正率
# y = to_categorical(y)
# fpr, tpr, thresholds = roc_curve(y.ravel(), y_pred.ravel())
#
# # 计算auc
# auc_logistics = auc(fpr, tpr)
#
# # 画图
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='knn(area={:.3f})'.format(auc_logistics))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.legend(loc='best')
# plt.show()


from sklearn.cluster import KMeans

from keras.datasets import mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train[:5000], x_test[:2000]))

y = np.concatenate((y_train[:5000], y_test[:2000]))

x = x.reshape((x.shape[0], -1))

x = np.divide(x, 255.)

# 10 clusters

n_clusters = len(np.unique(y))

# Runs in parallel 4 CPUs

kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)

# Train K-Means.

y_pred_kmeans = kmeans.fit_predict(x)

# Evaluate the K-Means clustering accuracy.

acc = accuracy_score(y, y_pred_kmeans)
print(acc)
