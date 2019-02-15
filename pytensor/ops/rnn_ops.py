from pytensor.network.tensor import *
from pytensor.network.parameter import *
from pytensor.network.operation import *
from pytensor.ops.rnn_util_ops import *

class RawRNNCell(Operation):

    def __init__(self,  name='RawRNNCell', argument=None, graph=None):

        super(RawRNNCell, self).__init__(name, argument, graph)

        # intialize size
        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        # batch size
        self.batch_size = 1

        # create tensors
        self.U = graph.parameter.get_tensor("RNNCell_U", (self.input_size, self.hidden_size))
        self.W = graph.parameter.get_tensor("RNNCell_W", (self.hidden_size, self.hidden_size))

        self.input_tensor = None
        self.state_tensor = None
        self.prev_state_tensor = None

    def forward(self, input_tensors):
        """
        forward computation of RNNCell
        State(t) = tanh(State(t-1)*W + Input(t)*U)

        :param input_tensors: input_tensors should contain 2 tensors: State(t-1) and Input(t)
        , State(t-1) is from previous cell and Input(t) is the current cell input tensor.
        :return: State(t)
        """

        # initialize State(t-1) if it is not provided
        if input_tensors[0] is None:
            self.batch_size = input_tensors[1].value.shape[0]

            prev_state_tensor = Tensor(np.zeros((self.batch_size, self.hidden_size)))
            input_tensors[0] = prev_state_tensor

        # forward registration
        self.register(input_tensors)

        # remember tensors
        self.prev_state_tensor = self.input_tensors[0]
        self.input_tensor = self.input_tensors[1]

        # input to hidden
        state_value = np.dot(self.input_tensor.value, self.U.value)

        # hidden to hidden
        state_value += np.dot(self.prev_state_tensor.value, self.W.value)

        # nonlinear
        state_value = np.tanh(state_value)

        # create tensor
        self.state_tensor = Tensor(state_value)
        return self.state_tensor

    def backward(self):
        """
        :param input_tensor: Input(t)
        :param state_tensor: State(t)
        :param loss_tensor: overall loss from State(t)
        :return: input_loss_tensor, prev_loss_tensor
        """

        # loss of State(t-1)*W+Input(t)*U
        dL = self.state_tensor.grad * (1.0 - self.state_tensor.value * self.state_tensor.value)

        dLdS = np.dot(dL, self.W.value.T)
        dLdW = np.dot(self.prev_state_tensor.value.T, dL)
        dLdI = np.dot(dL, self.U.value.T)
        dLdU = np.dot(self.input_tensor.value.T, dL)

        # add grad to U and W
        self.W.grad += dLdW
        self.U.grad += dLdU

        # update previous state's gradient
        self.prev_state_tensor.grad += dLdS

        # update input tensor's gradient
        self.input_tensor.grad += dLdI


class RNNCell(Operation):

    def __init__(self,  name, argument, graph):

        super(RNNCell, self).__init__(name, argument, graph)

        self.graph = graph

        # intialize size
        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        # batch size
        self.batch_size = 1

        self.rnn_affine = self.graph.get_operation("RNNAffine", {'input_size': self.input_size, 'hidden_size': self.hidden_size, "nonlinear": "Tanh"}, "RNNAffine")

        self.input_tensor = None

    def forward(self, input_tensors):
        """
        forward computation of RNNCell
        State(t) = tanh(State(t-1)*W + Input(t)*U)

        :param input_tensors: input_tensors should contain 2 tensors: State(t-1) and Input(t)
        , State(t-1) is from previous cell and Input(t) is the current cell input tensor.
        :return: State(t)
        """

        # initialize State(t-1) if it is not provided
        if input_tensors[0] is None:
            self.batch_size = input_tensors[1].value.shape[0]

            prev_state_tensor = Tensor(np.zeros((self.batch_size, self.hidden_size)))
            input_tensors[0] = prev_state_tensor

        # forward registration
        self.register(input_tensors)
        self.state_tensor = self.rnn_affine.forward(self.input_tensors)
        return self.state_tensor

    def backward(self):
        return

class RNN(Operation):

    def __init__(self, name='RNN', argument=None, graph=None):
        super(RNN, self).__init__(name, argument, graph)
        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        # max num steps to run RNN
        # this is to prevent initializing RNNCell everytime
        self.max_num_steps = argument['max_num_steps']
        self.num_steps = 0

        # create empty cell list
        self.cells = [RNNCell('RNNCell', argument, graph) for i in range(self.max_num_steps)]

        self.input_tensors = []
        self.state_tensors = []

    def forward(self, input_tensors):
        """

        :param input_tensors: a list of var_input from word embedding
        :return:
        """

        self.register(input_tensors)

        self.state_tensors = []

        self.num_steps = min(len(input_tensors), self.max_num_steps)

        self.last_state = None

        for i in range(self.num_steps):
            self.last_state = self.cells[i].forward([self.last_state, input_tensors[i]])
            self.state_tensors.append(self.last_state)

        return self.state_tensors

    def backward(self):
        return