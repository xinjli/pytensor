from network.base.operation import *

class Loss(Operation):

    def forward(self, input_variables):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def loss(self, target):
        raise NotImplementedError


class SoftmaxWithLoss(Loss):
    def __init__(self, name="SoftmaxWithLoss"):
        self.name = name

        self.input_variable = None
        self.output_variable = None
        self.param_variables = []

        self.target = None
        self.error = None
        self.batch_size = 0

    def forward(self, input_variable):

        self.input_variable = input_variable

        out_value = softmax(self.input_variable.value)
        self.output_variable = Variable(out_value)

        return self.output_variable

    def loss(self, target):

        self.batch_size = len(target)
        self.target = np.array(target)
        self.error = cross_entropy_error(self.output_variable.value, self.target)
        return self.error

    def backward(self):

        self.input_variable.grad = self.output_variable.value.copy()
        self.input_variable.grad[np.arange(self.batch_size), self.target] -= 1.0

        self.input_variable.grad *= 1.4426950408889634 # log2(e)


class SquareErrorLoss(Loss):

    def __init__(self, name="SquareErrorLoss"):
        self.name = name

    def forward(self, input_variable):
        self.input_variable = input_variable

    def loss(self, target):
        self.target = target
        loss_val =  mean_squared_error(self.input_variable.value, self.target.value)
        return loss_val

    def backward(self):
        self.input_variable.grad = self.input_variable.value - self.target.value
