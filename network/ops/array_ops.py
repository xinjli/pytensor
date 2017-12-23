from network.base.operation import *

class HStack:

    def __init__(self, name="HStack"):
        self.name = name

    def forward(self, input_variables):
        """
        Hstack  variables into one variable

        For instance, it can hstack [1,2], [3,4] into [1,2,3,4]

        :param input_variables:
        :return:
        """

        self.input_variables = input_variables
        self.input_cnt = len(self.input_variables)
        self.output_variable = Variable(np.hstack([input_variable.value for input_variable in self.input_variables]))

        return self.output_variable

    def backward(self):

        """
        backward output grad into each grad
        :return:
        """
        grads = np.hsplit(self.output_variable, self.input_cnt)

        for i, grad in enumerate(grads):
            self.input_variables[i].grad += grad


class VStack:

    def __init__(self, name="VStack"):
        self.name = name

    def forward(self, input_variables):
        """
        VStack variables into one variable

        :param input_variables:
        :return:
        """
        self.input_variables = input_variables
        self.input_cnt = len(self.input_variables)
        self.output_variable = Variable(np.vstack([input_variable.value for input_variable in self.input_variables]))

        return self.output_variable

    def backward(self):
        """
        :return:
        """

        grads = np.vsplit(self.output_variable.grad, self.input_cnt)

        for i, grad in enumerate(grads):
            self.input_variables[i].grad += np.squeeze(grad)
