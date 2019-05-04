# -*- coding: utf-8 -*-
# @Time: 19-4-18 下午4:23

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # 加载数据
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 加载5000条训练数据,和2000条测试数据
    train_images = train_images[:5000]
    train_labels = train_labels[:5000]
    test_images = test_images[:2000]
    test_labels = test_labels[:2000]


    # 准备图像数据(图像28*28,取值区间[0-1])
    train_images = train_images.reshape((5000, 28 * 28))
    train_images = train_images.astype("float32") / 255

    test_images = test_images.reshape((2000, 28 * 28))
    test_images = test_images.astype("float32") / 255

    # 准备标签
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # knn训练
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(train_images,train_labels)

    # 测试
    test_pred = knn.predict(test_images)

    # 获取准确度
    score = accuracy_score(test_pred, test_labels)

    # 获取真正率和假正率
    fpr, tpr, thresholds = roc_curve(test_labels.ravel(), test_pred.ravel())

    # 计算auc
    auc_logistics = auc(fpr, tpr)

    # 画图
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='knn(area={:.3f})'.format(auc_logistics))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    plt.show()
