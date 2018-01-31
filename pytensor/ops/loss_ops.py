from pytensor.network.operation import *

class Loss(Operation):
    def loss(self, target_variable):
        raise NotImplementedError


class SoftmaxLoss(Loss):
    def __init__(self, name="SoftmaxWithLoss", argument=None, graph=None):
        super(SoftmaxLoss, self).__init__(name, argument, graph)

    def forward(self, input_variables):
        super(SoftmaxLoss, self).forward(input_variables)

        out_value = softmax(self.input_variables.value)
        self.output_variable = Variable(out_value)

        return self.output_variable

    def loss(self, target_variable):

        self.target_variable = target_variable
        self.batch_size = len(self.target_variable.value)

        self.error = cross_entropy_error(self.output_variable.value, target_variable.value)
        return self.error

    def backward(self):

        self.input_variables.grad = self.output_variable.value.copy()
        self.input_variables.grad[np.arange(self.batch_size), self.target_variable.value] -= 1.0

        self.input_variables.grad *= 1.4426950408889634 # log2(e)


class SquareLoss(Loss):

    def __init__(self, name="SquareLoss", argument=None, graph=None):
        super(SquareLoss, self).__init__(name, argument, graph)

    def forward(self, input_variables):
        super(SquareLoss, self).forward(input_variables)

        out_value = softmax(self.input_variables.value)
        self.output_variable = Variable(out_value)

        return self.output_variable

    def loss(self, target_variable):
        self.target_variable = target_variable
        loss_val =  mean_squared_error(self.input_variables.value, self.target_variable.value)
        return loss_val

    def backward(self):
        # update grad
        self.input_variables.grad = self.input_variables.value - self.target_variable.value
