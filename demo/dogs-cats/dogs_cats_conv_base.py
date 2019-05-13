# -*- coding: utf-8 -*-
# @Author: wangchao
# @Time: 19-5-13 上午11:18
"""
使用预训练的卷积基提取特征
"""
import numpy as np
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# 处理图像
def extract_features(directory, sample_count):
    """
     预训练模型生成新模型输入数据
     :param directory: 源数据路径
     :param sample_count: 数据量
     :return: tuple
     """

    datagen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 20

    # 预训练生成数据(4,4,512)
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory=directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    # 使用预训练生成数据
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


# 构建模型
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])
    return model


# 绘制训练损失和验证损失
def paint_training_validation(history, type="loss"):
    """
    # 绘制训练和验证图
    :param history: 模型数据
    :param type: {loss:损失图,acc:精度图}
    :return: 
    """
    if type == "loss":
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training adn Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    else:
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # 预训练的卷积基
    conv_base = VGG16(weights="imagenet",
                      include_top=False,
                      input_shape=(150, 150, 3))

    train_dir = "/workspace/data/machine/dogs-cats/train"
    validation_dir = "/workspace/data/machine/dogs-cats/validation"
    test_dir = "/workspace/data/machine/dogs-cats/test"

    # train_dir = "/home/justyouso/space/data/machine/dogs-cats/train"
    # validation_dir = "/home/justyouso/space/data/machine/dogs-cats/validation"
    # test_dir = "/home/justyouso/space/data/machine/dogs-cats/test"

    # 获取提取特征
    train_features, train_labels = extract_features(train_dir, 2000)
    validation_features, validation_labels = extract_features(validation_dir,
                                                              1000)
    test_features, test_labels = extract_features(test_dir, 1000)

    # 特征平展
    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

    model = build_model()
    history = model.fit(
        train_features,
        train_labels,
        epochs=30,
        batch_size=20,
        validation_data=(validation_features, validation_labels)
    )
    model.save('cats_and_dogs_small_3.h5')
    # paint_training_validation(history)
    # paint_training_validation(history,"acc")

