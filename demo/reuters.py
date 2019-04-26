# -*- coding: utf-8 -*-
# @Author: wangchao
# @Time: 19-4-25 上午11:55
from keras.datasets import reuters
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    # 将第i行,第sequence列填充1
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results
