# -*- coding: utf-8 -*-
# @Time: 19-5-4 下午7:49
from keras.datasets import mnist
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# X shape (60,000 28x28), y shape (10,000, )
(x_train, _), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

# 压缩特征维度至2维
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(784,))

# 编码层
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# 解码层
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)

# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)

# compile autoencoder
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                metrics=["accuracy"])

# training
history = autoencoder.fit(x_train, x_train, epochs=20, batch_size=256,
                          shuffle=True,validation_data=(x_test,x_test))

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
# 绘制训练精度和验证精度
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# plotting
# encoded_imgs = encoder.predict(x_test)
# plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
# plt.colorbar()
# plt.show()
