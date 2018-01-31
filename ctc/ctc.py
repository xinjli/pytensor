from pytensor.network.operation import *

class Node:

    def __init__(self, frame_index, class_index):
        self.frame_index = frame_index
        self.class_index = class_index

        # viterbi search
        self.log_prob = 0.0
        self.best_prev = -1

        # alpha beta
        self.ln_alpha = -np.inf
        self.ln_beta = -np.inf


class SoftmaxCTCLoss:
    """
    This ctc computes all elems using log scale

    """

    def __init__(self):

        self.frames = []
        self.input_variables = []
        self.output_variables = []

        self.num_frame = 0
        self.num_class = 0
        self.num_label = 0

        # assume that blank index will always be indexed 0
        self.blank_index = 0

        # labels
        self.origin_labels = []
        self.labels = []

        # prob
        self.prob = 0.0

    def forward(self, input_variables):
        """
        input variables should be a list of variables whose values are prob distribution over classes

        :param input_variables:
        :return:
        """
        self.input_variables = input_variables

        self.num_frame = len(input_variables)
        self.num_class = len(input_variables[0].value)

        self.output_variables = []

        for input_variable in self.input_variables:
            output_val = softmax(input_variable.value)
            output_variable = Variable(output_val)

            self.output_variables.append(output_variable)

        return self.output_variables


    def loss(self, labels):

        # expand labels
        self.origin_labels = labels

        self.labels = [0]
        for c in self.origin_labels:
            self.labels.append(c)
            self.labels.append(0)

        self.num_label = len(self.labels)

        # check length
        assert(self.num_label <= self.num_frame)

        # initialize lattice
        self.lattice = [[Node(i, j) for j in range(self.num_label)] for i in range(self.num_frame)]

        # alpha and beta computation
        self._compute_alpha()
        self._compute_beta()

        # compute loss
        # sum over alpha beta using (7.26)
        # prob = P(z|x)
        self.ln_prob = -np.inf

        for i in range(self.num_label):
            ln_prod = elnprod(self.lattice[0][i].ln_alpha, self.lattice[0][i].ln_beta)
            self.ln_prob = elnsum(self.ln_prob, ln_prod)

        # loss = -log(P(z|x))
        return -self.ln_prob


    def _compute_alpha(self):

        # check whether probability is too low
        # assert(self.output_variables[0].value[self.labels[0]] >= 0.000001)
        # assert(self.output_variables[0].value[self.labels[1]] >= 0.000001)

        # initialize first row
        self.lattice[0][0].ln_alpha = eln(self.output_variables[0].value[self.labels[0]]) # for blank
        self.lattice[0][1].ln_alpha = eln(self.output_variables[0].value[self.labels[1]]) # for original first label

        # initialize first col
        for t in range(1, self.num_frame):
            self.lattice[t][0].ln_alpha = elnprod(self.lattice[t-1][0].ln_alpha, eln(self.output_variables[t].value[0]))

        # common alpha loop
        for t in range(1, self.num_frame):
            left_index = max(1, self.num_label-2*(self.num_frame-t))
            right_index = min(self.num_label, 2*(t+1))

            for i in range(left_index, right_index):

                label = self.labels[i]
                ln_y_prob = eln(self.output_variables[t].value[label])

                ln_alpha = elnsum(self.lattice[t-1][i].ln_alpha, self.lattice[t-1][i-1].ln_alpha)

                if label != self.blank_index and i >= 2 and label != self.labels[i-2]:
                    ln_alpha = elnsum(ln_alpha, self.lattice[t-1][i-2].ln_alpha)

                self.lattice[t][i].ln_alpha = elnprod(ln_alpha, ln_y_prob)


    def _compute_beta(self):

        last_frame = self.num_frame-1
        last_label = self.num_label-1

        # initialize last row
        self.lattice[last_frame][last_label].ln_beta = 0.0 # log(1)
        self.lattice[last_frame][last_label-1].ln_beta = 0.0 # log(1)

        # initialize last col
        for t in reversed(range(0, last_frame)):
            self.lattice[t][last_label].ln_beta = elnprod(self.lattice[t+1][last_label].ln_beta,
                                                          eln(self.output_variables[t+1].value[0])) # blank transitions

        # common beta loop
        for t in reversed(range(last_frame)):

            left_index = max(0, self.num_label-2*(self.num_frame-t))
            right_index = min(last_label, 2*(t+1))

            for i in range(left_index, right_index):

                ln_beta = elnprod(self.lattice[t+1][i].ln_beta, eln(self.output_variables[t+1].value[self.labels[i]]))
                ln_beta = elnsum(ln_beta, elnprod(self.lattice[t+1][i+1].ln_beta, eln(self.output_variables[t+1].value[self.labels[i+1]])))

                if self.labels[i] != 0 and i <= last_label-2 and self.labels[i] != self.labels[i+2]:
                    ln_beta = elnsum(ln_beta, elnprod(self.lattice[t+1][i+2].ln_beta, eln(self.output_variables[t+1].value[self.labels[i+2]])))

                self.lattice[t][i].ln_beta = ln_beta

    def backward(self):

        for t in range(self.num_frame):

            # compute sum over labels
            ln_label_prob = [-np.inf for i in range(self.num_class)]

            for i in range(self.num_label):
                label = self.labels[i]
                ln_label_prob[label] = elnsum(ln_label_prob[label], elnprod(self.lattice[t][i].ln_alpha, self.lattice[t][i].ln_beta))

            self.input_variables[t].grad = self.output_variables[t].value
            for i in range(self.num_class):
                self.input_variables[t].grad[i] -= np.exp(ln_label_prob[i]-self.ln_prob)

