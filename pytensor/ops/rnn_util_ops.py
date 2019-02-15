from pytensor.network.operation import *


class RNNAffine(Operation):

    def __init__(self, name="RNNAffine", argument=None, graph=None):
        """
        Affine transformation: y= nonlinear(U x1 + W x2 + b)
        This is a utility operation for RNN and LSTM

        :param argument: {'input_size', 'hidden_size', 'nonlinear'(optional), 'bias'(optional) }
        """

        super(RNNAffine, self).__init__(name, argument, graph)

        # one for input_size and the other for hidden_size
        assert ('input_size' in argument and 'hidden_size' in argument)

        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        self.graph = graph

        # bias is disabled
        if 'bias' in argument and argument['bias'] == 'None':
            self.b = None
        else:
            b_name = self.name + "_b"
            self.b = self.graph.parameter.get_tensor(b_name, (self.hidden_size,))

        # bias initialization
        if 'bias' in argument and isinstance(argument['bias'], float):
            self.b.value[::] = float(argument['bias'])

        # hidden to hidden
        self.U = self.graph.parameter.get_tensor(self.name+'_U', (self.hidden_size, self.hidden_size))

        # input to hidden
        self.W = self.graph.parameter.get_tensor(self.name+'_W', (self.input_size, self.hidden_size))

        # nonlinear operation
        self.nonlinear = None
        if 'nonlinear' in argument:
            nonlinear = argument['nonlinear']
            self.nonlinear = self.graph.get_operation(nonlinear, None, name+"_"+nonlinear)

        # output tensors
        self.add_tensor = None
        self.nonlinear_tensor = None


    def forward(self, input_tensors):
        self.register(input_tensors)

        # check input size
        assert input_tensors[1].value.shape[1] == self.input_size, "expected: " + str(
            self.input_size) + " actual: " + str(input_tensors[1].value.shape[1])

        # check input size
        assert input_tensors[0].value.shape[1] == self.hidden_size, "expected: " + str(
            self.input_size) + " actual: " + str(input_tensors[0].value.shape[1])

        # apply affine transformation
        value = np.dot(self.input_tensors[0].value, self.U.value) + np.dot(self.input_tensors[1].value, self.W.value)

        # add bias
        if self.b:
            value += self.b.value

        self.add_tensor = Tensor(value)

        if self.nonlinear:
            self.nonlinear_tensor = self.nonlinear.forward(self.add_tensor)
            return self.nonlinear_tensor
        else:
            return self.add_tensor

    def backward(self):

        self.input_tensors[1].grad += np.dot(self.add_tensor.grad, self.W.value.T)
        self.input_tensors[0].grad += np.dot(self.add_tensor.grad, self.U.value.T)

        self.W.grad += np.dot(self.input_tensors[1].value.T, self.add_tensor.grad)
        self.U.grad += np.dot(self.input_tensors[0].value.T, self.add_tensor.grad)

        if self.b:
            self.b.grad += np.sum(self.add_tensor.grad, axis=0)