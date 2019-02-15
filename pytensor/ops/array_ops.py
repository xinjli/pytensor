from pytensor.network.operation import *

class HStack:

    def __init__(self, name="HStack"):
        self.name = name

    def forward(self, input_tensors):
        """
        Hstack  tensors into one tensor

        For instance, it can hstack [1,2], [3,4] into [1,2,3,4]

        :param input_tensors:
        :return:
        """

        self.input_tensors = input_tensors
        self.input_cnt = len(self.input_tensors)
        self.output_tensor = Tensor(np.hstack([input_tensor.value for input_tensor in self.input_tensors]))

        return self.output_tensor

    def backward(self):

        """
        backward output grad into each grad
        :return:
        """
        grads = np.hsplit(self.output_tensor, self.input_cnt)

        for i, grad in enumerate(grads):
            self.input_tensors[i].grad += grad


class VStack:

    def __init__(self, name="VStack"):
        self.name = name

    def forward(self, input_tensors):
        """
        VStack tensors into one tensor

        :param input_tensors:
        :return:
        """
        self.input_tensors = input_tensors
        self.input_cnt = len(self.input_tensors)
        self.output_tensor = Tensor(np.vstack([input_tensor.value for input_tensor in self.input_tensors]))

        return self.output_tensor

    def backward(self):
        """
        :return:
        """

        grads = np.vsplit(self.output_tensor.grad, self.input_cnt)

        for i, grad in enumerate(grads):
            self.input_tensors[i].grad += np.squeeze(grad)
