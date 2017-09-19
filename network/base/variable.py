import numpy as np

class Variable:

    def __init__(self, value, name='Variable', trainable=True):
        """
        :param value: numpy val
        :param index:
        """
        self.value = np.array(value)
        self.grad = np.zeros(self.value.shape)

        self.name = name

        self.trainable = trainable

    def clear_grad(self):
        self.grad = np.zeros(self.value.shape)

    def reshape(self, array):
        self.value.reshape(array)
        self.grad.reshape(array)