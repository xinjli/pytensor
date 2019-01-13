from pytensor import *

class MLP:

    def __init__(self, input_size, hidden_size, output_size):
        self.graph = Graph("MLP")

        # make graph
        self.affine1 = self.graph.get_operation('Affine', {'input_size': input_size, 'hidden_size': hidden_size})
        self.sigmoid = self.graph.get_operation('Sigmoid')
        self.affine2 = self.graph.get_operation('Affine', {'input_size': hidden_size, 'hidden_size': output_size})
        self.softmaxloss = self.graph.get_operation('SoftmaxLoss')

    def forward(self, input_variable):
        affine1_variable = self.affine1.forward(input_variable)
        sigmoid_variable = self.sigmoid.forward(affine1_variable)
        affine2_variable = self.affine2.forward(sigmoid_variable)

        return self.softmaxloss.forward(affine2_variable)

    def loss(self, target_variable):
        return self.softmaxloss.loss(target_variable)
