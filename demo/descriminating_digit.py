# -*- coding: utf-8 -*-
# @Author: wangchao
# @Time: 19-4-18 下午4:23

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 网络架构
network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="sigmoid"))

# 编译训练模型
network.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                metrics=["accuracy"])

# 准备图像数据(图像28*28,取值区间[0-1])
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255

# 准备标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 测试模型
test_loss, test_acc = network.evaluate(test_images, test_labels)
