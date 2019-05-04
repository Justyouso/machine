# -*- coding: utf-8 -*-
# @Time: 19-4-25 上午11:55
from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    # 将第i行,第sequence列填充1
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


# 构建模型
def build_model():
    # 构建模型
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    # 编译模型
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
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
    # 获取reuters数据
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
        num_words=10000)

    # 向量化标签(,46),一个标签有46种分类
    one_hot_train_labels = to_one_hot(train_labels)
    one_hot_test_labels = to_one_hot(test_labels)

    # 向量化训练数据(,10000),一条评论有10000个单词组成
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # 留出验证集
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    # 构建并编译模型
    model = build_model()

    # 训练模型
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=9,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    # import copy
    # test_labels_copy = copy.copy(test_labels)
    # np.random.shuffle(test_labels_copy)
    # hits_array = np.array(test_labels) == np.array(test_labels_copy)

    # 绘制训练损失和验证损失
    paint_training_validation(history, "loss")
    # 绘制训练精度和验证精度
    paint_training_validation(history, "acc")

    # 预测数据
    predictions = model.predict(x_test)
    # 返回最大数的索引(46类中的一个类别)
    print(np.argmax(predictions[0]))
    print(test_labels[0])
