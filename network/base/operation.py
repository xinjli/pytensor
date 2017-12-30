from network.base.variable import *
from network.base.functions import *
from network.base.parameter import *

class Operation:
    def __init__(self, name, arguments, graph):

        self.name = name
        self.graph = graph
        self.arguments = arguments

    def __call__(self, input_variables):
        """
        shortcut method of forward

        :param input_variables:
        :return:
        """
        return self.forward(input_variables)


    def forward(self, input_variables):
        """
        forward computation

        :param input_variables: a list of input variables
        """

        self.input_variables = input_variables

        # register the ops
        if self.graph is not None:
            self.graph.register(self)

    def backward(self):
        """
        backprop loss and update

        :return:
        """
        raise NotImplementedError