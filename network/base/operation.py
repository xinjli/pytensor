from network.base.variable import *
from network.base.functions import *
from network.base.parameter import *

class Operation:
    """
    An interface that every operation should implement
    """

    def forward(self, input_variables):
        """
        forward computation

        :param input_variables: input variables
        :return: output variable
        """
        raise NotImplementedError

    def backward(self):
        """
        backprop loss and update

        :return:
        """
        raise NotImplementedError