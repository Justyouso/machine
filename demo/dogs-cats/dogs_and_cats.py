# -*- coding: utf-8 -*-
# @Author: wangchao
# @Time: 19-5-10 下午3:40

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 构建模型
def build_model():
    model = models.Sequential()
    model.add(
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    return model


# 处理图像
def images_del():
    train_dir = "/workspace/data/machine/dogs-cats/train"
    validation_dir = "/workspace/data/machine/dogs-cats/validation"

    # 缩放1/255倍
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # 训练数据生成器
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    # 验证数据生成器
    validation_generator = validation_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    return train_generator, validation_generator


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
    # 获取训练数据生成器和验证数据生成器
    train_generator,validation_generator = images_del()

    # 获取model
    model = build_model()

    # 训练模型
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )

    # paint_training_validation(history, 'acc')
    # 保存模型
    model.save('cats_and_dogs_small_1.h5')
