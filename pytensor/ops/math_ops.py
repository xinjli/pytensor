from pytensor.network.operation import *

class Add(Operation):

    def __init__(self, name='add', argument=None, graph=None):
        super(Add, self).__init__(name, argument, graph)


    def forward(self, input_variables):
        """
        Add all variables in the input_variable

        :param input_variables:
        :return:
        """
        super(Add, self).forward(input_variables)

        # value for the output variable
        value = np.zeros_like(self.input_variables[0])

        for input_variable in self.input_variables:
            value += input_variable.value

        self.output_variable = Variable(value)

        return self.output_variable

    def backward(self):
        """
        backward grad into each input variable

        :return:
        """

        for input_variable in self.input_variables:
            input_variable.grad += self.output_variable.grad


class Multiply(Operation):

    def __init__(self, name='multiply', argument=None, graph=None):
        super(Multiply, self).__init__(name, argument, graph)


    def forward(self, input_variables):
        """
        element multiplication of input_variables

        :param input_variables:
        :return:
        """
        super(Multiply, self).forward(input_variables)

        # only multiplication of two elements is supported now
        assert(len(input_variables)==2)

        self.input_variables = input_variables

        # validate both input_variables have same shape
        self.x = input_variables[0]
        self.y = input_variables[1]
        assert(self.x.shape == self.y.shape)

        # value for the output variable
        value = np.multiply(self.x, self.y)
        self.output_variable = Variable(value)

        return self.output_variable

    def backward(self):
        """
        backward grad into each input variables

        :return:
        """

        # update gradient
        self.x.grad += np.multiply(self.output_variable.grad, self.y.value)
        self.y.grad += np.multiply(self.output_variable.grad, self.x.value)


class Matmul(Operation):

    def __init__(self, name='matmul', argument=None, graph=None):
        super(Matmul, self).__init__(name, argument, graph)

    def forward(self, input_variables):
        super(Matmul, self).forward(input_variables)

        # only multiplication of two elements is supported now
        assert(len(input_variables)==2)

        self.x = input_variables[0]
        self.y = input_variables[1]

        out = np.dot(self.x.value, self.y.value)

        self.output_variable = Variable(out)
        return self.output_variable

    def backward(self):

        # update gradient
        self.x.grad += np.dot(self.output_variable.grad, self.y.value.T)
        self.y.grad += np.dot(self.x.value.T, self.output_variable.grad)


class Relu(Operation):
    def __init__(self, name="Relu", argument=None, graph=None):
        super(Relu, self).__init__(name, argument, graph)

        self.mask = None

    def forward(self, input_variables):
        """
        :param input_variable:
        :return:
        """

        super(Sigmoid, self).forward(input_variables)

        # compute mask
        self.mask = (input_variables.value <= 0)

        # create output variable
        out = input_variables.value.copy()
        out[self.mask] = 0
        self.output_variable = Variable(out)

        # return
        return self.output_variable

    def backward(self):
        self.input_variables.grad = self.output_variable.grad
        self.input_variables.grad[self.mask] = 0


class Sigmoid(Operation):

    def __init__(self, name='sigmoid', argument=None, graph=None):
        super(Sigmoid, self).__init__(name, argument, graph)

    def forward(self, input_variables):
        super(Sigmoid, self).forward(input_variables)

        # compute sigmoid
        value = sigmoid(self.input_variables.value)
        self.output_variable = Variable(value)

        return self.output_variable

    def backward(self):
        self.input_variables.grad += self.output_variable.grad * (1.0 - self.output_variable.value) * self.output_variable.value


class Tanh(Operation):

    def __init__(self, name='tanh', argument=None, graph=None):
        super(Tanh, self).__init__(name, argument, graph)

    def forward(self, input_variables):
        super(Tanh, self).forward(input_variables)

        self.input_variable = input_variables
        out = np.tanh(self.input_variable.value)

        self.output_variable = Variable(out)
        return self.output_variable

    def backward(self):
        self.input_variable.grad += self.output_variable.grad * (1.0 - self.output_variable.value * self.output_variable.value)


class Affine(Operation):
    def __init__(self, name="affine", argument=None, graph=None):
        """
        Affine transformation: y=wx+b

        :param argument: [input_size, hidden_size]
        """

        super(Affine, self).__init__(name, argument, graph)

        # arg should contains two int
        # one for input_size and the other for hidden_size
        assert(len(argument)==2)

        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        W_name = self.name + "_W"
        b_name = self.name + "_b"

        self.W = self.graph.parameter.get_variable(W_name, (self.input_size, self.hidden_size))
        self.b = self.graph.parameter.get_variable(b_name, (self.hidden_size, ))

    def forward(self, input_variables):
        super(Affine, self).forward(input_variables)

        # check input size
        assert(input_variables.value.shape[1] == self.input_size)

        # apply affine transformation
        value = np.dot(self.input_variables.value, self.W.value) + self.b.value
        self.output_variable = Variable(value)

        return self.output_variable

    def backward(self):
        self.input_variables.grad += np.dot(self.output_variable.grad, self.W.value.T)
        self.W.grad += np.dot(self.input_variables.value.T, self.output_variable.grad)
        self.b.grad += np.sum(self.output_variable.grad, axis=0)
