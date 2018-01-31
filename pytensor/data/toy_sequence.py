import numpy as np


def generate_toy_sequence(num):
    """
    Generate a toy sequence for debugging

    y = 2*x_1 + 3*x_2 + noise

    :param num: number of dataset
    :return: x, y
    """

    x = []
    y = []

    for i in range(num):

        new_x = np.array([[np.random.uniform(), np.random.uniform()]])
        new_y = np.array([[new_x[0][0]*2 + new_x[0][1]*3 + np.random.normal(0, scale=0.1)]])
        x.append(new_x)
        y.append(new_y)

    return x, y
