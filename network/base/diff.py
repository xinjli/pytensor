import numpy as np


def validate_gradient(self, loss_func, name):
    """
    validate numerical gradient with respect to a loss func

    :param loss_func:
    :param name: variable's name which we want to check gradient
    :return:
    """

    x = self.variable_dict[name].value

    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = loss_func()

        x[idx] = tmp_val - h
        fxh2 = loss_func()  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    print("Gradient ", grad)