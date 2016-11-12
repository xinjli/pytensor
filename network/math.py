import numpy as np

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.max(0, x)