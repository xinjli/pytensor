from pytensor.network.operation import *


class RNNAffine(Operation):

    def __init__(self, name="RNNAffine", argument=None, graph=None):
        """
        Affine transformation: y= nonlinear(W1 x1 + W2 x2)
        This is a utility operation for RNN and LSTM

        :param argument: {'input_size', 'hidden_size', 'nonlinear'(optional)}
        """

        super(RNNAffine, self).__init__(name, argument, graph)

        # arg should contains two int
        # one for input_size and the other for hidden_size
        assert (len(argument) == 2 or len(argument) ==3)
        assert ('input_size' in argument)
        assert ('hidden_size' in argument)

        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        self.graph = graph

        # create variables
        self.affine_U = self.graph.get_operation("Affine", {'input_size': self.input_size, 'hidden_size': self.hidden_size},  name+"_U")
        self.affine_W = self.graph.get_operation("Affine", {'input_size': self.hidden_size, 'hidden_size': self.hidden_size}, name+"_W")
        self.add = self.graph.get_operation("Add")

        # nonlinear operation
        self.nonlinear = None
        if 'nonlinear' in argument:
            nonlinear = argument['nonlinear']
            self.nonlinear = self.graph.get_operation(nonlinear, None, name+"_"+nonlinear)

    def forward(self, input_variables):
        super(RNNAffine, self).forward(input_variables)

        # remember variables
        self.prev_state_variable = self.input_variables[0]
        self.input_variable = self.input_variables[1]

        # input to hidden
        input_hidden_variable = self.affine_U.forward(self.input_variable)

        # hidden to hidden
        hidden_hidden_variable = self.affine_W.forward(self.prev_state_variable)

        # add two variables
        add_variable = self.add.forward([input_hidden_variable, hidden_hidden_variable])

        if self.nonlinear:
            self.state_variable = self.nonlinear.forward(add_variable)
            return self.state_variable
        else:
            return add_variable

    def backward(self):
        return