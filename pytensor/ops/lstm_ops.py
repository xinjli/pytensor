from pytensor.network.tensor import *
from pytensor.network.parameter import *
from pytensor.network.operation import *
from pytensor.ops.math_ops import *

class RawLSTMCell(Operation):

    def __init__(self,  name='RawLSTMCell', argument=None, graph=None):

        super(RawLSTMCell, self).__init__(name, argument, graph)

        # intialize size
        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        # batch size
        self.batch_size = 1

        # create tensors
        # forget gate weight
        self.Wf_h = graph.get_tensor(name+"LSTMCell_Wf_h", (self.hidden_size, self.hidden_size))
        self.Wf_i = graph.get_tensor(name+"LSTMCell_Wf_i", (self.input_size, self.hidden_size))

        # input gate weight
        self.Wi_h = graph.get_tensor(name+"LSTMCell_Wi_h", (self.hidden_size, self.hidden_size))
        self.Wi_i = graph.get_tensor(name+"LSTMCell_Wi_i", (self.input_size, self.hidden_size))

        # output gate weight
        self.Wo_h = graph.get_tensor(name+"LSTMCell_Wo_h", (self.hidden_size, self.hidden_size))
        self.Wo_i = graph.get_tensor(name+"LSTMCell_Wo_i", (self.input_size, self.hidden_size))

        # cell gate weight
        self.Wc_h = graph.get_tensor(name+"LSTMCell_Wc_h", (self.hidden_size, self.hidden_size))
        self.Wc_i = graph.get_tensor(name+"LSTMCell_Wc_i", (self.input_size, self.hidden_size))

        # tensors
        self.input_tensor = None

        # current state tensors
        self.hidden_state = None
        self.cell_state = None
        self.cell_input_state = None

        # previous state tensors
        self.prev_hidden_state = None
        self.prev_cell_state = None

        # forget, input, output
        self.forget_tensor = None
        self.in_tensor = None
        self.output_tensor = None
        self.cell_tensor = None
        self.cell_hidden_tensor = None

        self.forget_gate = Sigmoid(name="Forget_sigmoid")
        self.in_gate = Sigmoid(name="In_sigmoid")
        self.output_gate = Sigmoid(name="Output_sigmoid")
        self.cell_gate = Tanh(name="Cell_tanh")
        self.cell_hidden_gate = Tanh(name="Cell_hidden_tanh")

        self.forget_gate_tensor = None
        self.in_gate_tensor = None
        self.output_gate_tensor = None
        self.cell_gate_tensor = None


    def forward(self, input_tensors):
        """
        forward computation of LSTM

        f(t) = sigmoid(h(t-1)*Wf_h + input(t)*Wf_i)
        i(t) = sigmoid(h(t-1)*Wi_h + input(t)*Wi_i)
        o(t) = sigmoid(h(t-1)*Wo_h + input(t)*Wo_i)
        c(t) = tanh(h(t-1)*Wc_h + input(t)*Wc_i)

        cell(t) = f(t)*cell(t-1) * i(t)*c(t)
        hidden(t) = o(t)*tanh(cell(t))

        input_tensors should contain:
        - prev_hidden_state
        - prev_cell_state
        - input tensor

        """

        self.batch_size = input_tensors[2].value.shape[0]

        # initialize prev_hidden_state if not provided
        if input_tensors[0] is None:
            input_tensors[0] = Tensor(np.zeros((self.batch_size, self.hidden_size)))

        # initialize prev_cell_state if not provided
        if input_tensors[1] is None:
            input_tensors[1] = Tensor(np.zeros((self.batch_size, self.hidden_size)))

        self.register(input_tensors)

        # remember tensors
        self.prev_hidden_state = self.input_tensors[0]
        self.prev_cell_state = self.input_tensors[1]
        self.input_tensor = self.input_tensors[2]

        # compute forget gate
        # f(t) = sigmoid(h(t-1)*Wf_h + input(t)*Wf_i)
        self.forget_tensor = Tensor(np.dot(self.prev_hidden_state.value, self.Wf_h.value) + np.dot(self.input_tensor.value, self.Wf_i.value))
        self.forget_gate_tensor = self.forget_gate.forward(self.forget_tensor)

        # compute input gate
        # i(t) = sigmoid(h(t-1)*Wi_h + input(t)*Wi_i)
        self.in_tensor = Tensor(np.dot(self.prev_hidden_state.value, self.Wi_h.value) + np.dot(self.input_tensor.value, self.Wi_i.value))
        self.in_gate_tensor = self.in_gate.forward(self.in_tensor)

        # compute output gate
        # o(t) = sigmoid(h(t-1)*Wo_h + input(t)*Wo_i)
        self.output_tensor = Tensor(np.dot(self.prev_hidden_state.value, self.Wo_h.value) + np.dot(self.input_tensor.value, self.Wo_i.value))
        self.output_gate_tensor = self.output_gate.forward(self.output_tensor)

        # update cell gate
        # c(t) = tanh(h(t-1)*Wc_h + input(t)*Wc_i)
        # cell(t) = f(t)*cell(t-1) * i(t)*c(t)
        self.cell_tensor = Tensor(np.dot(self.prev_hidden_state.value, self.Wc_h.value) + np.dot(self.input_tensor.value, self.Wc_i.value))
        self.cell_gate_tensor = self.cell_gate.forward(self.cell_tensor)

        # compute current cell state
        # cell(t) = f(t)*cell(t-1) * i(t)*c(t)
        self.cell_state = Tensor(self.prev_cell_state.value * self.forget_gate_tensor.value +
                                   self.cell_gate_tensor.value * self.in_gate_tensor.value)

        # update hidden state
        # hidden(t) = o(t) * tanh(cell(t))
        self.cell_hidden_tensor = self.cell_hidden_gate.forward(self.cell_state)
        self.hidden_state = Tensor(self.output_gate_tensor.value * self.cell_hidden_tensor.value)

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
        self.output_gate_tensor.grad = self.hidden_state.grad * self.cell_hidden_tensor.value
        self.cell_hidden_tensor.grad = self.hidden_state.grad * self.output_gate_tensor.value
        self.cell_hidden_gate.backward()

        # update following gradient
        # cell(t) = f(t)*cell(t-1) + i(t)*c(t)
        self.prev_cell_state.grad = self.cell_state.grad * self.forget_gate_tensor.value
        self.forget_gate_tensor.grad = self.cell_state.grad * self.prev_cell_state.value
        self.in_gate_tensor.grad = self.cell_state.grad * self.cell_gate_tensor.value
        self.cell_gate_tensor.grad = self.cell_state.grad * self.in_gate_tensor.value

        # update following gradient
        # c(t) = tanh(h(t-1)*Wc_h + input(t)*Wc_i)
        self.cell_gate.backward()
        self.Wc_h.grad += np.outer(self.prev_hidden_state.value, self.cell_tensor.grad)
        self.Wc_i.grad += np.outer(self.input_tensor.value, self.cell_tensor.grad)

        self.prev_hidden_state.grad += np.dot(self.cell_tensor.grad, self.Wc_h.value.T)
        self.input_tensor.grad += np.dot(self.cell_tensor.grad, self.Wc_i.value.T)

        # update following gradient
        # o(t) = sigmoid(h(t-1)*Wo_h + input(t)*Wo_i)
        self.output_gate.backward()
        self.Wo_h.grad += np.outer(self.prev_hidden_state.value, self.output_tensor.grad)
        self.Wo_i.grad += np.outer(self.input_tensor.value, self.output_tensor.grad)

        self.prev_hidden_state.grad += np.dot(self.output_tensor.grad, self.Wo_h.value.T)
        self.input_tensor.grad += np.dot(self.output_tensor.grad, self.Wo_i.value.T)

        # update following gradient
        # i(t) = sigmoid(h(t-1)*Wi_h + input(t)*Wi_i)
        self.in_gate.backward()
        self.Wi_h.grad += np.outer(self.prev_hidden_state.value, self.in_tensor.grad)
        self.Wi_i.grad += np.outer(self.input_tensor.value, self.in_tensor.grad)

        self.prev_hidden_state.grad += np.dot(self.in_tensor.grad, self.Wi_h.value.T)
        self.input_tensor.grad += np.dot(self.in_tensor.grad, self.Wi_i.value.T)

        # update following gradient
        # f(t) = sigmoid(h(t-1)*Wf_h + input(t)*Wf_i)
        self.forget_gate.backward()
        self.Wf_h.grad += np.outer(self.prev_hidden_state.value, self.forget_tensor.grad)
        self.Wf_i.grad += np.outer(self.input_tensor.value, self.forget_tensor.grad)

        self.prev_hidden_state.grad += np.dot(self.forget_tensor.grad, self.Wf_h.value.T)
        self.input_tensor.grad += np.dot(self.forget_tensor.grad, self.Wf_i.value.T)

class LSTMCell(Operation):

    def __init__(self,  name='LSTMCell', argument=None, graph=None):

        super(LSTMCell, self).__init__(name, argument, graph)

        # intialize size
        self.input_size = argument['input_size']
        self.hidden_size = argument['hidden_size']

        # batch size
        self.batch_size = 1

        self.graph = graph

        # set forget bias to 1 to prevent gradient vanishing
        self.forget_gate  = self.graph.get_operation("RNNAffine", {'input_size': self.input_size, 'hidden_size': self.hidden_size, "nonlinear": "Sigmoid", "bias": 1.0}, "LSTMForget")
        self.input_gate   = self.graph.get_operation("RNNAffine", {'input_size': self.input_size, 'hidden_size': self.hidden_size, "nonlinear": "Sigmoid"}, "LSTMInput")
        self.output_gate  = self.graph.get_operation("RNNAffine", {'input_size': self.input_size, 'hidden_size': self.hidden_size, "nonlinear": "Sigmoid"}, "LSTMOutput")
        self.cell_gate    = self.graph.get_operation("RNNAffine", {'input_size': self.input_size, 'hidden_size': self.hidden_size, "nonlinear": "Tanh"}, "LSTMCell")
        self.forget_multi = self.graph.get_operation("Multiply")
        self.input_multi  = self.graph.get_operation("Multiply")
        self.output_multi = self.graph.get_operation("Multiply")
        self.tanh         = self.graph.get_operation("Tanh")
        self.add          = self.graph.get_operation("Add")

        # tensors
        self.input_tensor = None

        # current state tensors
        self.hidden_state = None
        self.cell_state = None

        # previous state tensors
        self.prev_hidden_state = None
        self.prev_cell_state = None

        # forget, input, output
        self.forget_state = None
        self.input_state = None
        self.output_state = None
        self.hidden_tensor = None
        self.cell_hidden_tensor = None

        self.forget_gate_tensor = None
        self.in_gate_tensor = None
        self.output_gate_tensor = None
        self.cell_gate_tensor = None


    def forward(self, input_tensors):
        """
        forward computation of LSTM

        f(t) = sigmoid(h(t-1)*Wf_h + input(t)*Wf_i)
        i(t) = sigmoid(h(t-1)*Wi_h + input(t)*Wi_i)
        o(t) = sigmoid(h(t-1)*Wo_h + input(t)*Wo_i)
        c(t) = tanh(h(t-1)*Wc_h + input(t)*Wc_i)

        cell(t) = f(t)*cell(t-1) * i(t)*c(t)
        hidden(t) = o(t)*tanh(cell(t))

        input_tensors should contain:
        - prev_hidden_state
        - prev_cell_state
        - input tensor

        """

        self.batch_size = input_tensors[2].value.shape[0]

        # initialize prev_hidden_state if not provided
        if input_tensors[0] is None:
            input_tensors[0] = Tensor(np.zeros((self.batch_size, self.hidden_size)))

        # initialize prev_cell_state if not provided
        if input_tensors[1] is None:
            input_tensors[1] = Tensor(np.zeros((self.batch_size, self.hidden_size)))

        self.register(input_tensors)

        # remember tensors
        self.prev_hidden_state = self.input_tensors[0]
        self.prev_cell_state = self.input_tensors[1]
        self.input_tensor = self.input_tensors[2]
        input_hidden_pair = [self.prev_hidden_state, self.input_tensor]

        self.forget_gate_tensor = self.forget_gate.forward(input_hidden_pair)
        self.input_gate_tensor = self.input_gate.forward(input_hidden_pair)
        self.output_gate_tensor = self.output_gate.forward(input_hidden_pair)
        self.cell_gate_tensor = self.cell_gate.forward(input_hidden_pair)

        # compute current cell state
        # cell(t) = f(t)*cell(t-1) + i(t)*c(t)
        self.forget_state = self.forget_multi.forward([self.forget_gate_tensor, self.prev_cell_state])
        self.input_state  = self.input_multi.forward([self.input_gate_tensor, self.cell_gate_tensor])
        self.cell_state   = self.add.forward([self.forget_state, self.input_state])

        # update hidden state
        # hidden(t) = o(t) * tanh(cell(t))
        self.cell_hidden_tensor = self.tanh.forward(self.cell_state)
        self.hidden_state = self.output_multi.forward([self.output_gate_tensor, self.cell_hidden_tensor])

        # return hidden and cell
        return self.hidden_state, self.cell_state


    def backward(self):
        return


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

        self.input_tensors = []
        self.state_tensors = []

    def forward(self, input_tensors):
        """

        :param input_tensors: a list of var_input from word embedding
        :return:
        """

        self.register(input_tensors)

        self.hidden_states = []
        self.cell_states = []

        self.num_steps = min(len(input_tensors), self.max_num_steps)

        self.last_hidden = None
        self.last_cell = None

        for i in range(self.num_steps):
            self.last_hidden, self.last_cell = self.cells[i].forward([self.last_hidden, self.last_cell, input_tensors[i]])
            self.hidden_states.append(self.last_hidden)
            self.cell_states.append(self.last_cell)

        return self.hidden_states

    def backward(self):
        return