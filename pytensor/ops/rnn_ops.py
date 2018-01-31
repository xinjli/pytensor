from pytensor.network.variable import *
from pytensor.network.parameter import *
from pytensor.network.operation import *

class RNNCell(Operation):

    def __init__(self,  name='RNNCell', argument=None, graph=None):

        super(RNNCell, self).__init__(name, argument, graph)

        # intialize size
        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        # batch size
        self.batch_size = 1

        # create variables
        self.U = graph.parameter.get_variable("RNNCell_U", (self.input_size, self.hidden_size))
        self.W = graph.parameter.get_variable("RNNCell_W", (self.hidden_size, self.hidden_size))

        self.input_variable = None
        self.state_variable = None
        self.prev_state_variable = None

    def forward(self, input_variables):
        """
        forward computation of RNNCell
        State(t) = tanh(State(t-1)*W + Input(t)*U)

        :param input_variables: input_variables should contain 2 variables: State(t-1) and Input(t)
        , State(t-1) is from previous cell and Input(t) is the current cell input variable.
        :return: State(t)
        """

        # initialize State(t-1) if it is not provided
        if input_variables[0] is None:
            self.batch_size = input_variables[1].value.shape[0]

            prev_state_variable = Variable(np.zeros((self.batch_size, self.hidden_size)))
            input_variables[0] = prev_state_variable

        # forward registration
        super(RNNCell, self).forward(input_variables)

        # remember variables
        self.prev_state_variable = self.input_variables[0]
        self.input_variable = self.input_variables[1]

        # input to hidden
        state_value = np.dot(self.input_variable.value, self.U.value)

        # hidden to hidden
        state_value += np.dot(self.prev_state_variable.value, self.W.value)

        # nonlinear
        state_value = np.tanh(state_value)

        # create variable
        self.state_variable = Variable(state_value)
        return self.state_variable

    def backward(self):
        """
        :param input_variable: Input(t)
        :param state_variable: State(t)
        :param loss_variable: overall loss from State(t)
        :return: input_loss_variable, prev_loss_variable
        """

        # loss of State(t-1)*W+Input(t)*U
        dL = self.state_variable.grad * (1.0 - self.state_variable.value * self.state_variable.value)

        dLdS = np.dot(dL, self.W.value.T)
        dLdW = np.dot(self.prev_state_variable.value.T, dL)
        dLdI = np.dot(dL, self.U.value.T)
        dLdU = np.dot(self.input_variable.value.T, dL)

        # add grad to U and W
        self.W.grad += dLdW
        self.U.grad += dLdU

        # update previous state's gradient
        self.prev_state_variable.grad += dLdS

        # update input variable's gradient
        self.input_variable.grad += dLdI


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

        self.input_variables = []
        self.state_variables = []

    def forward(self, input_variables):
        """

        :param input_variables: a list of var_input from word embedding
        :return:
        """

        super(RNN, self).forward(input_variables)

        self.state_variables = []

        self.num_steps = min(len(input_variables), self.max_num_steps)

        self.last_state = None

        for i in range(self.num_steps):
            self.last_state = self.cells[i].forward([self.last_state, input_variables[i]])
            self.state_variables.append(self.last_state)

        return self.state_variables

    def backward(self):
        return