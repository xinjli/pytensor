from network.base.variable import *
from network.base.parameter import *
from network.base.operation import *
from network.ops.math_ops import *

class LSTMCell(Operation):

    def __init__(self,  name='RNNCell', argument=None, graph=None):

        super(LSTMCell, self).__init__(name, argument, graph)

        # intialize size
        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        # batch size
        self.batch_size = 1

        # create variables
        # forget gate weight
        self.Wf_h = graph.get_variable(name+"LSTMCell_Wf_h", (self.hidden_size, self.hidden_size))
        self.Wf_i = graph.get_variable(name+"LSTMCell_Wf_i", (self.input_size, self.hidden_size))

        # input gate weight
        self.Wi_h = graph.get_variable(name+"LSTMCell_Wi_h", (self.hidden_size, self.hidden_size))
        self.Wi_i = graph.get_variable(name+"LSTMCell_Wi_i", (self.input_size, self.hidden_size))

        # output gate weight
        self.Wo_h = graph.get_variable(name+"LSTMCell_Wo_h", (self.hidden_size, self.hidden_size))
        self.Wo_i = graph.get_variable(name+"LSTMCell_Wo_i", (self.input_size, self.hidden_size))

        # cell gate weight
        self.Wc_h = graph.get_variable(name+"LSTMCell_Wc_h", (self.hidden_size, self.hidden_size))
        self.Wc_i = graph.get_variable(name+"LSTMCell_Wc_i", (self.input_size, self.hidden_size))

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


    def forward(self, input_variables):
        """
        forward computation of LSTM

        f(t) = sigmoid(h(t-1)*Wf_h + input(t)*Wf_i)
        i(t) = sigmoid(h(t-1)*Wi_h + input(t)*Wi_i)
        o(t) = sigmoid(h(t-1)*Wo_h + input(t)*Wo_i)
        c(t) = tanh(h(t-1)*Wc_h + input(t)*Wc_i)

        cell(t) = f(t)*cell(t-1) * i(t)*c(t)
        hidden(t) = o(t)*tanh(cell(t))

        input_variables should contain:
        - prev_hidden_state
        - prev_cell_state
        - input variable

        """

        self.batch_size = input_variables[2].value.shape[0]

        # initialize prev_hidden_state if not provided
        if input_variables[0] is None:
            input_variables[0] = Variable(np.zeros((self.batch_size, self.hidden_size)))

        # initialize prev_cell_state if not provided
        if input_variables[1] is None:
            input_variables[1] = Variable(np.zeros((self.batch_size, self.hidden_size)))


        super(LSTMCell, self).forward(input_variables)

        # remember variables
        self.prev_hidden_state = self.input_variables[0]
        self.prev_cell_state = self.input_variables[1]
        self.input_variable = self.input_variables[2]

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


class LSTM(Operation):

    def __init__(self, name='LSTM', argument=None, graph=None):
        super(LSTM, self).__init__(name, argument, graph)
        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        # max num steps to run RNN
        # this is to prevent initializing RNNCell everytime
        self.max_num_steps = argument['max_num_steps']
        self.num_steps = 0

        # create empty cell list
        self.cells = [LSTMCell('LSTMCell', argument, graph) for i in range(self.max_num_steps)]

        self.input_variables = []
        self.state_variables = []

    def forward(self, input_variables):
        """

        :param input_variables: a list of var_input from word embedding
        :return:
        """

        super(LSTM, self).forward(input_variables)

        self.hidden_states = []
        self.cell_states = []

        self.num_steps = min(len(input_variables), self.max_num_steps)

        self.last_hidden = None
        self.last_cell = None

        for i in range(self.num_steps):
            self.last_hidden, self.last_cell = self.cells[i].forward([self.last_hidden, self.last_cell, input_variables[i]])
            self.hidden_states.append(self.last_hidden)
            self.cell_states.append(self.last_cell)

        return self.hidden_states

    def backward(self):
        return