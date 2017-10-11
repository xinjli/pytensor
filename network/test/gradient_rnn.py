from network.base.rnn import *
from network.base.parameter import *
from network.base.operation import *
from network.base.optimizer import *
from network.base.gradient import *
from network.base.gradient import *
from common.logger import *

import numpy as np


class TestGradientRNN:

    def __init__(self):

        self.num_steps = 1

        input_lst = [ Variable(np.random.randn(3), trainable=False) for i in range(self.num_steps) ]
        output_lst = [ np.random.randn(2) for i in range(self.num_steps) ]
        self.parameter = Parameter()

        self.rnn = RNN(3, 2, 100, self.parameter)
        self.errors = [SquareErrorLoss() for i in range(100) ]

        self.optimizer = SGD(self.parameter)

    def train(self, input_lst, output_lst):

        for ii in range(1):
            loss_sum = 0.0

            variables = self.rnn.forward(self.input_lst)
            for i in range(self.num_steps):
                self.errors[i].forward(variables[i])
                loss_sum += self.errors[i].get_loss(self.output_lst[i])
                self.errors[i].backward()

            self.rnn.backward()
            self.optimizer.update()

            print("Loss sum ", loss_sum)

    def forward(self, input_lst):

        self.input_lst = input_lst
        self.num_steps = len(input_lst)

        variables = self.rnn.forward(self.input_lst)
        for i in range(self.num_steps):
            self.errors[i].forward(variables[i])

    def loss(self, output_lst):

        loss_sum = 0.0
        for i in range(self.num_steps):
            loss_sum += self.errors[i].get_loss(output_lst[i])

        return loss_sum

    def backward(self):

        for i in range(self.num_steps):
            self.errors[i].backward()

        self.rnn.backward()

    def get_numerical_gradient(self):

        print("Numerical Gradient RNNCell_U")
        #grad = numerical_gradient(self.get_loss, self.param.variable_dict['RNNCell_U'].value)
        self.parameter.validate_gradient(self.loss, 'RNNCell_W')

        self.train()


if __name__ == '__main__':
    rnntest = TestGradientRNN()

    num_steps = 20
    input_lst = [Variable(np.random.randn(3), trainable=False) for i in range(num_steps)]
    output_lst = [np.random.randn(2) for i in range(num_steps)]

    validate_gradient(rnntest, input_lst, output_lst)
