from pytensor.network.tensor import *
from pytensor.network.functions import *
from pytensor.network.parameter import *

class Operation:
    def __init__(self, name, arguments, graph):

        self.name = name
        self.graph = graph
        self.arguments = arguments

    def __call__(self, input_tensors):
        """
        shortcut method of forward

        :param input_tensors:
        :return:
        """
        return self.forward(input_tensors)


    def register(self, input_tensors):
        """
        forward computation

        :param input_tensors: a list of input tensors
        """

        self.input_tensors = input_tensors

        # register the ops
        if self.graph is not None:
            self.graph.register(self)

    def forward(self, input_tensors):

        raise NotImplementedError


    def backward(self):
        """
        backprop loss and update

        :return:
        """
        raise NotImplementedError