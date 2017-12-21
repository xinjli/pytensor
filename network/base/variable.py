import numpy as np

class Variable:
    """
    Variable is the basic structure in the computation graph
    It holds value for forward computation and grad for

    """

    def __init__(self, value, name='Variable', trainable=True):
        """
        :param value: numpy val
        :param name: name for the variable
        :param trainable: whether the variable can be trained or not
        """

        # value for forward computation
        self.value = np.array(value)

        # value for backward computation
        self.grad = np.zeros(self.value.shape)

        self.name = name

        self.trainable = trainable

    def clear_grad(self):
        self.grad = np.zeros(self.value.shape)

    def reshape(self, array):
        self.value.reshape(array)
        self.grad.reshape(array)