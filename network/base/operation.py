from network.base.variable import *
from network.base.functions import *
from network.base.parameter import *

class HStack:

    def __init__(self, name="HStack"):
        self.name = name

    def forward(self, input_variables):
        """
        Hstack  variables into one variable

        For instance, it can hstack [1,2], [3,4] into [1,2,3,4]

        :param input_variables:
        :return:
        """

        self.input_variables = input_variables
        self.input_cnt = len(self.input_variables)
        self.output_variable = Variable(np.hstack([input_variable.value for input_variable in self.input_variables]))

        return self.output_variable

    def backward(self):

        """
        backward output grad into each grad
        :return:
        """
        grads = np.hsplit(self.output_variable, self.input_cnt)

        for i, grad in enumerate(grads):
            self.input_variables[i].grad += grad


class VStack:

    def __init__(self, name="VStack"):
        self.name = name

    def forward(self, input_variables):
        """
        VStack variables into one variable

        :param input_variables:
        :return:
        """
        self.input_variables = input_variables
        self.input_cnt = len(self.input_variables)
        self.output_variable = Variable(np.vstack([input_variable.value for input_variable in self.input_variables]))

        return self.output_variable

    def backward(self):
        """
        :return:
        """

        grads = np.vsplit(self.output_variable.grad, self.input_cnt)

        for i, grad in enumerate(grads):
            self.input_variables[i].grad += np.squeeze(grad)



class Relu:
    def __init__(self, name="Relu"):
        self.name = name
        self.mask = None

    def forward(self, input_variable):
        """
        :param input_variable:
        :return:
        """

        # remember input variable
        self.input_variable = input_variable

        # compute mask
        self.mask = (input_variable.value <= 0)

        # create output variable
        out = input_variable.value.copy()
        out[self.mask] = 0
        self.output_variable = Variable(out)

        # return
        return self.output_variable

    def backward(self):
        self.input_variable.grad = self.output_variable.grad
        self.input_variable.grad[self.mask] = 0


class Sigmoid:
    def __init__(self, name='Sigmoid'):
        self.name = name
        self.input_variable = None
        self.output_variable = None

    def forward(self, input_variable):
        self.input_variable = input_variable
        out = sigmoid(self.input_variable.value)

        self.output_variable = Variable(out)
        return self.output_variable

    def backward(self):
        self.input_variable.grad += self.output_variable.grad * (1.0 - self.output_variable.value) * self.output_variable.value


class Tanh:

    def __init__(self, name="Tanh"):
        self.name = name
        self.input_variable = None
        self.output_variable = None

    def forward(self, input_variable):
        self.input_variable = input_variable
        out = np.tanh(self.input_variable.value)

        self.output_variable = Variable(out)
        return self.output_variable

    def backward(self):
        self.input_variable.grad += self.output_variable.grad * (1.0 - self.output_variable.value * self.output_variable.value)


class Affine:
    def __init__(self, input_size, hidden_size, parameter, name='Affine'):
        self.name = name

        self.input_size = input_size
        self.hidden_size = hidden_size

        W_name = self.name + "_W"
        b_name = self.name + "_b"

        self.W = parameter.get_variable(W_name, (input_size, hidden_size))
        self.b = parameter.get_variable(b_name, (hidden_size, ))

    def forward(self, input_variable):
        self.input_variable = input_variable
        out = np.dot(self.input_variable.value, self.W.value) + self.b.value

        self.output_variable = Variable(out)
        return self.output_variable

    def backward(self):
        self.input_variable.grad += np.dot(self.output_variable.grad, self.W.value.T)
        self.W.grad += np.dot(self.input_variable.value.T, self.output_variable.grad)
        self.b.grad += np.sum(self.output_variable.grad, axis=0)


class SoftmaxWithLoss:
    def __init__(self, name="SoftmaxWithLoss"):
        self.name = name

        self.input_variable = None
        self.output_variable = None
        self.param_variables = []

        self.target = None
        self.error = None
        self.batch_size = 0

    def forward(self, input_variable):

        self.input_variable = input_variable

        out_value = softmax(self.input_variable.value)
        self.output_variable = Variable(out_value)

        return self.output_variable

    def loss(self, target):

        self.batch_size = len(target)
        self.target = np.array(target)
        self.error = cross_entropy_error(self.output_variable.value, self.target)
        return self.error

    def backward(self):

        self.input_variable.grad = self.output_variable.value.copy()
        self.input_variable.grad[np.arange(self.batch_size), self.target] -= 1.0

        self.input_variable.grad *= 1.4426950408889634 # log2(e)


class SquareErrorLoss:

    def __init__(self, name="SquareErrorLoss"):
        self.name = name

    def forward(self, input_variable):
        self.input_variable = input_variable

    def loss(self, target):
        self.target = np.array(target)
        return mean_squared_error(self.input_variable.value, self.target)

    def backward(self):
        self.input_variable.grad = self.input_variable.value - self.target
