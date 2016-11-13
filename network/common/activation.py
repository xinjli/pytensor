import numpy as np


def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.max(0, x)

