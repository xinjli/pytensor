from network.base.optimizer import *
from network.base.parameter import *
from collections import defaultdict

from network.ops.math_ops import *
from network.ops.loss_ops import *
from network.ops.array_ops import *


class Graph:

    def __init__(self, name):
        """
        graph is a data structure to manage parameter and execution order

        """

        # graph name
        self.name = name

        # forward operation queue
        self.forward_ops = []

        # parameter and optimizer
        self.parameter = Parameter()
        self.optimizer = SGD(self.parameter)

        # index for different operations
        self.ops_index = defaultdict(int)

    def get_operation(self, ops_type, ops_argument=None):
        """
        create operation

        :return:

        """
        # generate the default name
        ops_name = ops_type.lower()+"_"+str(self.ops_index[ops_type])

        # increment index
        self.ops_index[ops_type] += 1

        # reflection
        cls = globals()[ops_type]
        ops = cls(ops_name, ops_argument, self)

        return ops

    def forward(self, ops):
        """
        register a forward ops to the queue (for backward computation)

        :param ops:
        :return:
        """

        self.forward_ops.append(ops)


    def clear(self):
        """
        clear operation queue

        :return:
        """

        self.forward_ops = []


    def backward(self):
        """
        backward the error in reverse topological order

        :return:
        """

        for ops in reversed(self.forward_ops):
            ops.backward()

        # clear all operation
        self.clear()