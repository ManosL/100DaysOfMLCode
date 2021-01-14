import numpy as np
import math
import random

from isolationTree import IsolationTree

euler_constant = 0.5772156649

class IsolationForest:
    def __init__(self, trees_no=100, sample_size=256):
        self.__trees_no     = trees_no
        self.__sample_size  = sample_size
        self.__height_limit = math.ceil(math.log2(sample_size))
        self.__trees        = []

    def fit(self, X): # X is a numpy array
        self.__trees = []
        for _ in range(self.__trees_no):
            sample_indexes = random.sample(list(range(len(X))), self.__sample_size)
            X_sample       = X[sample_indexes, :]

            self.__trees.append(IsolationTree(X_sample, self.__height_limit))

        return

    def predict(self, X):
        predictions = []

        for i in range(len(X)):
            sample           = X[i]
            path_lengths_sum = 0

            for tree in self.__trees:
                path_lengths_sum += tree.pathLength(sample)

            path_lengths_avg = path_lengths_sum / self.__trees_no

            c_num  = 2 * (math.log(self.__sample_size - 1) + euler_constant)
            c_num -= 2 * ((self.__sample_size - 1) / self.__sample_size)

            prediction = 2 ** (-1 * (path_lengths_avg / c_num))
            
            assert (prediction >= 0.0) and (prediction <= 1.0)

            predictions.append(prediction)

        return np.array(predictions)