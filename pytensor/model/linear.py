from pytensor.network.graph import Graph
from pytensor.network.variable import Variable

class Linear(Graph):

    def __init__(self, input_size, output_size):
        super().__init__("linear")

        # make graph
        self.affine = self.get_operation('Affine', {'input_size' : input_size, 'hidden_size': output_size})
        self.softmaxloss = self.get_operation('SoftmaxLoss')

    def forward(self, input_variable):
        affine_variable = self.affine.forward(input_variable)
        return self.softmaxloss.forward(affine_variable)

    def loss(self, target_variable):
        return self.softmaxloss.loss(target_variable)
