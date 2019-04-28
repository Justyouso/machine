# -*- coding: utf-8 -*-
# @Author: wangchao
# @Time: 19-4-26 下午2:00
from keras.datasets import boston_housing
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt


def build_model():
    model = models.Sequential()
    model.add(
        layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point(1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def k_validation():
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []
    for i in range(k):
        print('processing fold #', i)
        # 验证数据
        val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
        val_targets = train_targets[
                      i * num_val_samples:(i + 1) * num_val_samples]

        partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                             train_data[
                                                 (i + 1) * num_val_samples]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples]], axis=0)


if __name__ == "__main__":
    (train_data, train_targets), (
        test_data, test_targets) = boston_housing.load_data()
    # 标准化测试数据(x-x^)/均方差
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    # K折验证
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []
    all_mae_histories =[]
    for i in range(k):
        print('processing fold #', i)
        # 验证数据
        val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
        val_targets = train_targets[
                      i * num_val_samples:(i + 1) * num_val_samples]

        partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                             train_data[
                                                 (i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]], axis=0)
        # 构建模型
        model = build_model()
        history = model.fit(partial_train_data,
                  partial_train_targets,
                  validation_data=(val_data,val_targets),
                  epochs=num_epochs,
                  batch_size=1, verbose=0)
        all_mae_histories.append(history.history['val_mean_absolute_error'])
    # 求出k个分区100次迭代的平均值
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in
                           range(num_epochs)]

    # 画图
    plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    #     all_scores.append(val_mae)i
