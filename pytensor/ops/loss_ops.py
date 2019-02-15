from pytensor.network.operation import *

class Loss(Operation):
    def loss(self, target_tensor):
        raise NotImplementedError


class SoftmaxLoss(Loss):
    def __init__(self, name="SoftmaxWithLoss", argument=None, graph=None):
        super(SoftmaxLoss, self).__init__(name, argument, graph)

    def forward(self, input_tensors):
        self.register(input_tensors)

        out_value = softmax(self.input_tensors.value)
        self.output_tensor = Tensor(out_value)

        return self.output_tensor

    def loss(self, target_tensor):

        self.target_tensor = target_tensor
        self.batch_size = len(self.target_tensor.value)

        self.error = cross_entropy_error(self.output_tensor.value, target_tensor.value)
        return self.error

    def backward(self):

        self.input_tensors.grad = self.output_tensor.value.copy()
        self.input_tensors.grad[np.arange(self.batch_size), self.target_tensor.value] -= 1.0

        self.input_tensors.grad *= 1.4426950408889634 # log2(e)


class SquareLoss(Loss):

    def __init__(self, name="SquareLoss", argument=None, graph=None):
        super(SquareLoss, self).__init__(name, argument, graph)

    def forward(self, input_tensors):
        self.register(input_tensors)

        out_value = softmax(self.input_tensors.value)
        self.output_tensor = Tensor(out_value)

        return self.output_tensor

    def loss(self, target_tensor):
        self.target_tensor = target_tensor
        loss_val =  mean_squared_error(self.input_tensors.value, self.target_tensor.value)
        return loss_val

    def backward(self):
        # update grad
        self.input_tensors.grad = self.input_tensors.value - self.target_tensor.value
