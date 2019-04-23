from pytensor.network.graph import Graph
from pytensor.network.tensor import Tensor, LongTensor

class Linear(Graph):

    def __init__(self, input_size, output_size):
        super().__init__("linear")

        # make graph
        self.affine = self.get_operation('Affine', {'input_size' : input_size, 'hidden_size': output_size})
        self.softmaxloss = self.get_operation('SoftmaxLoss')

    def forward(self, input_tensor):
        affine_tensor = self.affine.forward(input_tensor)
        return self.softmaxloss.forward(affine_tensor)

    def loss(self, target_tensor):
        return self.softmaxloss.loss(target_tensor)
