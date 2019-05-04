# -*- coding: utf-8 -*-
# @Time: 19-4-18 下午6:08

from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[10]
print(1,train_images)
train_images = train_images[:,2:4,2:4]
print(2,train_images)
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()