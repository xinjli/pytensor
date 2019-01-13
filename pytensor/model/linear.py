from pytensor.network.graph import Graph
from pytensor.network.variable import Variable

class Linear:

    def __init__(self, input_size, output_size):

        self.graph = Graph("Linear")

        # make graph
        self.affine = self.graph.get_operation('Affine', {'input_size' : input_size, 'hidden_size': output_size})
        self.softmaxloss = self.graph.get_operation('SoftmaxLoss')

    def forward(self, input_variable):
        affine_variable = self.affine.forward(input_variable)
        return self.softmaxloss.forward(affine_variable)

    def loss(self, target_variable):
        return self.softmaxloss.loss(target_variable)
