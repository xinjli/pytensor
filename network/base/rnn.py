from network.base.variable import *
from network.base.parameter import *

class RNNCell:

    def __init__(self,  input_size, hidden_size, parameter):

        # intialize size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # batch size
        self.batch_size = 1

        # create variables
        self.U = parameter.get_variable("RNNCell_U", (self.input_size, self.hidden_size))
        self.W = parameter.get_variable("RNNCell_W", (self.hidden_size, self.hidden_size))

        self.input_variable = None
        self.state_variable = None
        self.prev_state_variable = None

    def forward(self, input_variable, prev_state_variable=None):
        """

        forward computation of RNNCell
        State(t) = tanh(State(t-1)*W + Input(t)*U)

        :param input_variable: Variable from embedding
        :param state_variable: State(t-1) variable
        :return: State(t)
        """

        # remember variables
        self.input_variable = input_variable
        self.prev_state_variable = prev_state_variable

        self.batch_size = self.input_variable.value.shape[0]

        # check prev state variable
        if self.prev_state_variable == None:
            self.prev_state_variable = Variable(np.zeros((self.batch_size, self.hidden_size)))

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


class RNN:

    def __init__(self, input_size, hidden_size, max_num_steps, parameter):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # max num steps to run RNN
        # this is to prevent initializing RNNCell everytime
        self.max_num_steps = max_num_steps
        self.num_steps = 0

        # create empty cell list
        self.cells = [RNNCell(self.input_size, self.hidden_size, parameter) for i in range(self.max_num_steps)]

        self.input_variables = []
        self.state_variables = []

    def forward(self, input_variables):
        """

        :param input_variables: a list of var_input from word embedding
        :return:
        """

        self.input_variables = input_variables
        self.state_variables = []

        self.num_steps = min(len(input_variables), self.max_num_steps)

        self.last_state = None

        for i in range(self.num_steps):
            self.last_state = self.cells[i].forward(input_variables[i], self.last_state)
            self.state_variables.append(self.last_state)

        return self.state_variables


    def backward(self):
        """
        :param loss_variables: loss variables dLds
        :return:
        """

        dLdS = np.zeros(self.hidden_size)

        for i in reversed(range(self.num_steps)):
            self.cells[i].backward()