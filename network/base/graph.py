from network.ops.factory import *
from network.base.optimizer import *
from network.base.parameter import *


class Graph:

    def __init__(self, name):
        """
        graph is a structure to manage parameter and execution order

        """

        # graph name
        self.name = name

        # executed operation
        self.ops = []
        self.ops_num = 0

        # parameter and optimizer
        self.parameter = Parameter()
        self.optimizer = SGD(self.parameter)


    def get_operation(self, ops_type, ops_name=None, ops_argument=None):
        """
        create operation

        :return:

        """

        ops = create_operation(ops_type, ops_name, ops_argument, self)
        return ops


    def register(self, ops):
        """
        register an executed ops

        :param ops:
        :return:
        """

        self.ops.append(ops)
        self.ops_num += 1

    def backward(self):
        """
        backward the error in reverse topological order

        :return:
        """

        for ops in reversed(self.ops):
            ops.backward()
