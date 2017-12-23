import numpy as np

class Variable:
    """
    Variable is the basic structure in the computation graph
    It holds value for forward computation and grad for

    """

    def __init__(self, value, name='Variable', input_ops=None, trainable=True):
        """
        :param value: numpy val
        :param name: name for the variable
        :param trainable: whether the variable can be trained or not
        """

        # value for forward computation
        self.value = np.array(value)

        # value for backward computation
        self.grad = np.zeros(self.value.shape)

        # name for the variable (which will used in parameter for registration)
        self.name = name

        # whether the variable can be updated
        self.trainable = trainable

        # the operation that produced this variable
        self.input_ops = input_ops

        # the number of operation that depends on this variable
        self.dependency_cnt = 0


    def clear_grad(self):
        self.grad = np.zeros(self.value.shape)

    def reshape(self, array):
        self.value.reshape(array)
        self.grad.reshape(array)

    def backward(self):
        """
        back propagation gradient into previous operation if it received all grad from dependency operations

        :return:
        """

        # no backprop if no dependency
        if self.input_ops is None:
            return

        self.dependency_cnt -= 1
        if self.dependency_cnt <= 0:
            self.input_ops.backward()