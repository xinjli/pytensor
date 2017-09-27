from network.base.variable import *
from network.base.parameter import *
from network.base.operation import *

class LSTMCell:

    def __init__(self,  input_size, hidden_size, parameter, name=''):

        # initialize size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # forget gate weight
        self.Wf_h = parameter.get_variable(name+"LSTMCell_Wf_h", (self.hidden_size, self.hidden_size))
        self.Wf_i = parameter.get_variable(name+"LSTMCell_Wf_i", (self.input_size, self.hidden_size))

        # input gate weight
        self.Wi_h = parameter.get_variable(name+"LSTMCell_Wi_h", (self.hidden_size, self.hidden_size))
        self.Wi_i = parameter.get_variable(name+"LSTMCell_Wi_i", (self.input_size, self.hidden_size))

        # output gate weight
        self.Wo_h = parameter.get_variable(name+"LSTMCell_Wo_h", (self.hidden_size, self.hidden_size))
        self.Wo_i = parameter.get_variable(name+"LSTMCell_Wo_i", (self.input_size, self.hidden_size))

        # cell gate weight
        self.Wc_h = parameter.get_variable(name+"LSTMCell_Wc_h", (self.hidden_size, self.hidden_size))
        self.Wc_i = parameter.get_variable(name+"LSTMCell_Wc_i", (self.input_size, self.hidden_size))

        # variables
        self.input_variable = None

        # current state variables
        self.hidden_state = None
        self.cell_state = None
        self.cell_input_state = None

        # previous state variables
        self.prev_hidden_state = None
        self.prev_cell_state = None

        # forget, input, output
        self.forget_variable = None
        self.in_variable = None
        self.output_variable = None
        self.cell_variable = None
        self.cell_hidden_variable = None

        self.forget_gate = Sigmoid(name="Forget_sigmoid")
        self.in_gate = Sigmoid(name="In_sigmoid")
        self.output_gate = Sigmoid(name="Output_sigmoid")
        self.cell_gate = Tanh(name="Cell_tanh")
        self.cell_hidden_gate = Tanh(name="Cell_hidden_tanh")

        self.forget_gate_variable = None
        self.in_gate_variable = None
        self.output_gate_variable = None
        self.cell_gate_variable = None


    def forward(self, input_variable, prev_hidden_state=None, prev_cell_state=None):
        """
        forward computation of LSTM

        f(t) = sigmoid(h(t-1)*Wf_h + input(t)*Wf_i)
        i(t) = sigmoid(h(t-1)*Wi_h + input(t)*Wi_i)
        o(t) = sigmoid(h(t-1)*Wo_h + input(t)*Wo_i)
        c(t) = tanh(h(t-1)*Wc_h + input(t)*Wc_i)

        cell(t) = f(t)*cell(t-1) * i(t)*c(t)
        hidden(t) = o(t)*tanh(cell(t))
        """

        # remember variables
        self.input_variable = input_variable
        self.prev_hidden_state = prev_hidden_state
        self.prev_cell_state = prev_cell_state

        # check prev state variable
        if self.prev_hidden_state == None:
            self.prev_hidden_state = Variable(np.zeros(self.hidden_size))

        # check prev cell variable
        if self.prev_cell_state == None:
            self.prev_cell_state = Variable(np.zeros(self.hidden_size))

        # compute forget gate
        # f(t) = sigmoid(h(t-1)*Wf_h + input(t)*Wf_i)
        self.forget_variable = Variable(np.dot(self.prev_hidden_state.value, self.Wf_h.value) + np.dot(self.input_variable.value, self.Wf_i.value))
        self.forget_gate_variable = self.forget_gate.forward(self.forget_variable)

        # compute input gate
        # i(t) = sigmoid(h(t-1)*Wi_h + input(t)*Wi_i)
        self.in_variable = Variable(np.dot(self.prev_hidden_state.value, self.Wi_h.value) + np.dot(self.input_variable.value, self.Wi_i.value))
        self.in_gate_variable = self.in_gate.forward(self.in_variable)

        # compute output gate
        # o(t) = sigmoid(h(t-1)*Wo_h + input(t)*Wo_i)
        self.output_variable = Variable(np.dot(self.prev_hidden_state.value, self.Wo_h.value) + np.dot(self.input_variable.value, self.Wo_i.value))
        self.output_gate_variable = self.output_gate.forward(self.output_variable)

        # update cell gate
        # c(t) = tanh(h(t-1)*Wc_h + input(t)*Wc_i)
        # cell(t) = f(t)*cell(t-1) * i(t)*c(t)
        self.cell_variable = Variable(np.dot(self.prev_hidden_state.value, self.Wc_h.value) + np.dot(self.input_variable.value, self.Wc_i.value))
        self.cell_gate_variable = self.cell_gate.forward(self.cell_variable)

        # compute current cell state
        # cell(t) = f(t)*cell(t-1) * i(t)*c(t)
        self.cell_state = Variable(self.prev_cell_state.value * self.forget_gate_variable.value +
                                   self.cell_gate_variable.value * self.in_gate_variable.value)

        # update hidden state
        # hidden(t) = o(t) * tanh(cell(t))
        self.cell_hidden_variable = self.cell_hidden_gate.forward(self.cell_state)
        self.hidden_state = Variable(self.output_gate_variable.value * self.cell_hidden_variable.value)

        # return hidden and cell
        return self.hidden_state, self.cell_state


    def backward(self):
        """
        LSTM backward computation

        self.hidden_state.grad should be computed in advance by higher layers
        self.cell.grad should also be set if current element is not the last element

        """

        # update gradient of following formulas
        # hidden(t) = o(t)*tanh(cell(t))
        self.output_gate_variable.grad = self.hidden_state.grad * self.cell_hidden_variable.value
        self.cell_hidden_variable.grad = self.hidden_state.grad * self.output_gate_variable.value
        self.cell_hidden_gate.backward()

        # update following gradient
        # cell(t) = f(t)*cell(t-1) + i(t)*c(t)
        self.prev_cell_state.grad = self.cell_state.grad * self.forget_gate_variable.value
        self.forget_gate_variable.grad = self.cell_state.grad * self.prev_cell_state.value
        self.in_gate_variable.grad = self.cell_state.grad * self.cell_gate_variable.value
        self.cell_gate_variable.grad = self.cell_state.grad * self.in_gate_variable.value

        # update following gradient
        # c(t) = tanh(h(t-1)*Wc_h + input(t)*Wc_i)
        self.cell_gate.backward()
        self.Wc_h.grad += np.outer(self.prev_hidden_state.value, self.cell_variable.grad)
        self.Wc_i.grad += np.outer(self.input_variable.value, self.cell_variable.grad)

        self.prev_hidden_state.grad += np.dot(self.cell_variable.grad, self.Wc_h.value.T)
        self.input_variable.grad += np.dot(self.cell_variable.grad, self.Wc_i.value.T)

        # update following gradient
        # o(t) = sigmoid(h(t-1)*Wo_h + input(t)*Wo_i)
        self.output_gate.backward()
        self.Wo_h.grad += np.outer(self.prev_hidden_state.value, self.output_variable.grad)
        self.Wo_i.grad += np.outer(self.input_variable.value, self.output_variable.grad)

        self.prev_hidden_state.grad += np.dot(self.output_variable.grad, self.Wo_h.value.T)
        self.input_variable.grad += np.dot(self.output_variable.grad, self.Wo_i.value.T)

        # update following gradient
        # i(t) = sigmoid(h(t-1)*Wi_h + input(t)*Wi_i)
        self.in_gate.backward()
        self.Wi_h.grad += np.outer(self.prev_hidden_state.value, self.in_variable.grad)
        self.Wi_i.grad += np.outer(self.input_variable.value, self.in_variable.grad)

        self.prev_hidden_state.grad += np.dot(self.in_variable.grad, self.Wi_h.value.T)
        self.input_variable.grad += np.dot(self.in_variable.grad, self.Wi_i.value.T)

        # update following gradient
        # f(t) = sigmoid(h(t-1)*Wf_h + input(t)*Wf_i)
        self.forget_gate.backward()
        self.Wf_h.grad += np.outer(self.prev_hidden_state.value, self.forget_variable.grad)
        self.Wf_i.grad += np.outer(self.input_variable.value, self.forget_variable.grad)

        self.prev_hidden_state.grad += np.dot(self.forget_variable.grad, self.Wf_h.value.T)
        self.input_variable.grad += np.dot(self.forget_variable.grad, self.Wf_i.value.T)


class LSTM:

    def __init__(self, input_size, hidden_size, max_num_steps, parameter):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # max num steps to run RNN
        # this is to prevent initializing RNNCell everytime
        self.max_num_steps = max_num_steps
        self.num_steps = 0

        # create empty cell list
        self.cells = [LSTMCell(self.input_size, self.hidden_size, parameter) for i in range(self.max_num_steps)]

        self.input_variables = []
        self.hidden_states = []
        self.cell_states = []

    def forward(self, input_variables):
        """

        :param input_variables: a list of var_input from word embedding
        :return:
        """

        self.input_variables = input_variables
        self.hidden_states = []
        self.cell_states = []

        self.num_steps = min(len(input_variables), self.max_num_steps)

        self.last_hidden = None
        self.last_cell = None

        for i in range(self.num_steps):
            self.last_hidden, self.last_cell = self.cells[i].forward(input_variables[i], self.last_hidden, self.last_cell)
            self.hidden_states.append(self.last_hidden)
            self.cell_states.append(self.last_cell)

        return self.hidden_states

    def backward(self):
        """
        :param loss_variables: loss variables dLds
        :return:
        """
        for i in reversed(range(self.num_steps)):
            self.cells[i].backward()


class BLSTM:

    def __init__(self, input_size, hidden_size, max_num_steps, parameter):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # max num steps to run RNN
        # this is to prevent initializing RNNCell everytime
        self.max_num_steps = max_num_steps
        self.num_steps = 0

        # create empty cell list
        self.forward_cells = [LSTMCell(self.input_size, self.hidden_size, parameter, 'BLSTM Forward/') for i in range(self.max_num_steps)]
        self.backward_cells = [LSTMCell(self.input_size, self.hidden_size, parameter, 'BLSTM Backward/') for i in range(self.max_num_steps)]

        self.input_variables = []
        self.forward_hidden_states = []
        self.backward_hidden_states = []

        # combine forward and backward hidden
        self.Wf = parameter.get_variable("BLSTM Forward/Output", (self.hidden_size, self.hidden_size))
        self.Wb = parameter.get_variable("BLSTM Backward/Output", (self.hidden_size, self.hidden_size))

        # final output
        self.output_combined_variables = []
        self.output_sigmoid = [Sigmoid(name='BLSTM Sigmoid/Output') for i in range(self.max_num_steps)]
        self.output_variables = []


    def forward(self, input_variables):
        """

        :param input_variables: a list of var_input from word embedding
        :return:
        """

        self.input_variables = input_variables

        self.forward_hidden_states = []
        self.backward_hidden_states = []

        self.forward_cell_states = []
        self.backward_cell_states = []

        self.num_steps = min(len(input_variables), self.max_num_steps)

        self.last_forward_hidden = None
        self.last_backward_hidden = None

        self.last_forward_cell = None
        self.last_backward_cell = None

        # run forward path
        for i in range(self.num_steps):
            self.last_forward_hidden, self.last_forward_cell = self.forward_cells[i].forward(input_variables[i], self.last_forward_hidden, self.last_forward_cell)
            self.forward_hidden_states.append(self.last_forward_hidden)

        # run backward path
        for i in reversed(range(self.num_steps)):
            self.last_backward_hidden, self.last_backward_cell = self.backward_cells[i].forward(input_variables[i], self.last_backward_hidden, self.last_backward_cell)
            self.backward_hidden_states.append(self.last_backward_hidden)



        # combine forward and backward
        # output = sigmoid(Forward[t]*Wf + Backward[t]*Wb)
        self.output_variables = []


        for i in range(self.num_steps):
            output_combined_variable = Variable(np.dot(self.forward_hidden_states[i].value, self.Wf.value) + np.dot(self.backward_hidden_states[i].value, self.Wb.value))
            output_variable = self.output_sigmoid[i].forward(output_combined_variable)

            self.output_combined_variables.append(output_combined_variable)
            self.output_variables.append(output_variable)


        return self.output_variables


    def backward(self):
        """
        :param loss_variables: loss variables dLds
        :return:
        """

        # split errors into forward and backward
        for i in range(self.num_steps):
            self.output_sigmoid[i].backward()
            output_combined_variable = self.output_combined_variables[i]

            self.forward_hidden_states[i].grad += np.dot(output_combined_variable.grad, self.Wf.value.T)
            self.backward_hidden_states[i].grad += np.dot(output_combined_variable.grad, self.Wb.value.T)

            self.Wf.grad += np.outer(self.forward_hidden_states[i].value, output_combined_variable.grad)
            self.Wb.grad += np.outer(self.backward_hidden_states[i].value, output_combined_variable.grad)


        # run backward for Forward path
        for i in reversed(range(self.num_steps)):
           self.forward_cells[i].backward()

        # run backward for Backward path
        for i in range(self.num_steps):
            self.backward_cells[i].backward()
