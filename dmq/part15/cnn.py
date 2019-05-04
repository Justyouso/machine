# -*- coding: utf-8 -*-
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
network.add(layers.Dense(10, activation="softmax"))

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
history =network.fit(train_images, train_labels, epochs=5, batch_size=128,
            validation_data=(test_images, test_labels))



# 取出精度
history_dict = history.history

loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1,len(loss_values)+1)

# 绘制训练损失和验证损失
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 清空图像
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# 测试模型
# test_loss, test_acc = network.evaluate(test_images, test_labels)
