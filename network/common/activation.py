import numpy as np


def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.max(0, x)

