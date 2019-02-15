import numpy as np

class LongTensor:
    """
    LongTensor is a type of Tensor to keep integers

    """

    def __init__(self, value, name='LongTensor', trainable=False):
        """
        :param value: long value
        :param name:
        :param trainable:
        """

        self.value = np.array(value, dtype='int64')
        self.name = name

    def clear_grad(self):
        return

    def reshape(self, array):
        return


class Tensor:
    """
    Tensor is the basic structure in the computation graph
    It holds value for forward computation and grad for backward propagation

    """

    def __init__(self, value, name='Tensor', trainable=True):
        """
        :param value: numpy val
        :param name: name for the Tensor
        :param trainable: whether the Tensor can be trained or not
        """

        # value for forward computation
        self.value = np.array(value)

        # value for backward computation
        self.grad = np.zeros(self.value.shape)

        # name for the Tensor (which will used in parameter for registration)
        self.name = name

        # whether the Tensor can be updated
        self.trainable = trainable

    def __str__(self):
        return "Tensor {name: "+self.name+"}\n- value    : "+str(self.value)+"\n- gradient : "+str(self.grad)+""

    def __repr__(self):
        return self.__str__()


    def clear_grad(self):
        self.grad = np.zeros(self.value.shape)

    def reshape(self, array):
        self.value.reshape(array)
        self.grad.reshape(array)
