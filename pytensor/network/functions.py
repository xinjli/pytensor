# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # for preventing overflow
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    """
    mean square error

    :param y:
    :param t:
    :return:
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):

    # add dimension if it is not batch
    #if y.ndim == 1:
    #    y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log2(y[np.arange(batch_size), int(t)]))


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
