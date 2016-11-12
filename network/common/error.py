import numpy as np


def mean_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)


def cross_entropy_error(y, t):
    """
    :param y:
    :param t:
    :return:
    """

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y)) / batch_size

def label_cross_entropy_error(y, t):

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
