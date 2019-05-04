# -*- coding: utf-8 -*-
# @Author: wangchao
# @Time: 19-4-18 下午4:23

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras import models


from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc


def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / P


# tensorflow的评估函数Auc(可用作模型评估)
def tf_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# def auc(y_true, y_pred):
#     ptas = tf.stack(
#         [binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)],
#         axis=0)
#     pfas = tf.stack(
#         [binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)],
#         axis=0)
#     pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
#     binSizes = -(pfas[1:] - pfas[:-1])
#     s = ptas * binSizes
#     return K.sum(s, axis=0)

if __name__ == "__main__":
    # 加载数据
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 加载5000条训练数据,和2000条测试数据
    train_images = train_images[:5000]
    train_labels = train_labels[:5000]
    test_images = test_images[:2000]
    test_labels = test_labels[:2000]
    # 网络架构
    network = models.Sequential()
    network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation="softmax"))

    # 编译训练模型
    network.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                    metrics=[tf_auc])

    # 准备图像数据(图像28*28,取值区间[0-1])
    train_images = train_images.reshape((5000, 28 * 28))
    train_images = train_images.astype("float32") / 255

    test_images = test_images.reshape((2000, 28 * 28))
    test_images = test_images.astype("float32") / 255

    # 准备标签
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # 训练模型
    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    # 测试模型
    test_pred = network.predict(test_images)

    # # 获取真正率和假正率
    fpr, tpr, thresholds = roc_curve(test_labels.ravel(), test_pred.ravel())
    # 计算auc
    auc_logistics = auc(fpr, tpr)

    # 画图
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='logistics(area={:.3f})'.format(auc_logistics))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    plt.show()

    # 测试模型
    # test_loss, test_acc = network.evaluate(test_images, test_labels)
    # print(test_loss, test_acc)
