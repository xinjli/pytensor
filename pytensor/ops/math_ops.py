from pytensor.network.operation import *

class Add(Operation):

    def __init__(self, name='add', argument=None, graph=None):
        super(Add, self).__init__(name, argument, graph)


    def forward(self, input_tensors):
        """
        Add all tensors in the input_tensor

        :param input_tensors:
        :return:
        """
        self.register(input_tensors)

        # value for the output tensor
        value = np.zeros_like(self.input_tensors[0].value)

        for input_tensor in self.input_tensors:
            value += input_tensor.value

        self.output_tensor = Tensor(value)

        return self.output_tensor

    def backward(self):
        """
        backward grad into each input tensor

        :return:
        """

        for input_tensor in self.input_tensors:
            input_tensor.grad += self.output_tensor.grad


class Multiply(Operation):

    def __init__(self, name='multiply', argument=None, graph=None):
        super(Multiply, self).__init__(name, argument, graph)


    def forward(self, input_tensors):
        """
        element multiplication of input_tensors

        :param input_tensors:
        :return:
        """
        self.register(input_tensors)

        # only multiplication of two elements is supported now
        assert(len(input_tensors)==2)

        self.input_tensors = input_tensors

        # validate both input_tensors have same shape
        self.x = input_tensors[0]
        self.y = input_tensors[1]
        assert(self.x.value.shape == self.y.value.shape)

        # value for the output tensor
        value = np.multiply(self.x.value, self.y.value)
        self.output_tensor = Tensor(value)

        return self.output_tensor

    def backward(self):
        """
        backward grad into each input tensors

        :return:
        """

        # update gradient
        self.x.grad += np.multiply(self.output_tensor.grad, self.y.value)
        self.y.grad += np.multiply(self.output_tensor.grad, self.x.value)


class Matmul(Operation):

    def __init__(self, name='matmul', argument=None, graph=None):
        super(Matmul, self).__init__(name, argument, graph)

    def forward(self, input_tensors):
        self.register(input_tensors)

        # only multiplication of two elements is supported now
        assert(len(input_tensors)==2)

        self.x = input_tensors[0]
        self.y = input_tensors[1]

        out = np.dot(self.x.value, self.y.value)

        self.output_tensor = Tensor(out)
        return self.output_tensor

    def backward(self):

        # update gradient
        self.x.grad += np.dot(self.output_tensor.grad, self.y.value.T)
        self.y.grad += np.dot(self.x.value.T, self.output_tensor.grad)


class Relu(Operation):
    def __init__(self, name="Relu", argument=None, graph=None):
        super(Relu, self).__init__(name, argument, graph)

        self.mask = None

    def forward(self, input_tensors):
        """
        :param input_tensor:
        :return:
        """

        self.register(input_tensors)

        # compute mask
        self.mask = (input_tensors.value <= 0)

        # create output tensor
        out = input_tensors.value.copy()
        out[self.mask] = 0
        self.output_tensor = Tensor(out)

        # return
        return self.output_tensor

    def backward(self):
        self.input_tensors.grad = self.output_tensor.grad
        self.input_tensors.grad[self.mask] = 0


class Sigmoid(Operation):

    def __init__(self, name='sigmoid', argument=None, graph=None):
        super(Sigmoid, self).__init__(name, argument, graph)

    def forward(self, input_tensors):
        self.register(input_tensors)

        # compute sigmoid
        value = sigmoid(self.input_tensors.value)
        self.output_tensor = Tensor(value)

        return self.output_tensor

    def backward(self):
        self.input_tensors.grad += self.output_tensor.grad * (1.0 - self.output_tensor.value) * self.output_tensor.value


class Tanh(Operation):

    def __init__(self, name='tanh', argument=None, graph=None):
        super(Tanh, self).__init__(name, argument, graph)

    def forward(self, input_tensors):
        self.register(input_tensors)

        self.input_tensor = input_tensors
        out = np.tanh(self.input_tensor.value)

        self.output_tensor = Tensor(out)
        return self.output_tensor

    def backward(self):
        self.input_tensor.grad += self.output_tensor.grad * (1.0 - self.output_tensor.value * self.output_tensor.value)


class Affine(Operation):
    def __init__(self, name="affine", argument=None, graph=None):
        """
        Affine transformation: y=wx+b

        :param argument: [input_size, hidden_size, bias (optional)]
        """

        super(Affine, self).__init__(name, argument, graph)

        # arg should contains two int
        # one for input_size and the other for hidden_size
        assert(len(argument)==2 or len(argument)==3)

        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        # bias is disabled
        if 'bias' in argument and argument['bias'] == 'None':
            self.b = None
        else:
            b_name = self.name + "_b"
            self.b = self.graph.parameter.get_tensor(b_name, (self.hidden_size,))

        # bias initialization
        if 'bias' in argument and isinstance(argument['bias'], float):
            self.b.value[::] = float(argument['bias'])

        W_name = self.name + "_W"
        self.W = self.graph.parameter.get_tensor(W_name, (self.input_size, self.hidden_size))


    def forward(self, input_tensors):
        self.register(input_tensors)

        # check input size
        assert input_tensors.value.shape[1] == self.input_size, "expected: "+str(self.input_size)+" actual: "+str(input_tensors.value.shape[1])

        # apply affine transformation
        value = np.dot(self.input_tensors.value, self.W.value)

        # add bias
        if self.b:
            value += self.b.value

        self.output_tensor = Tensor(value)
        return self.output_tensor

    def backward(self):
        self.input_tensors.grad += np.dot(self.output_tensor.grad, self.W.value.T)
        self.W.grad += np.dot(self.input_tensors.value.T, self.output_tensor.grad)

        if self.b:
            self.b.grad += np.sum(self.output_tensor.grad, axis=0)