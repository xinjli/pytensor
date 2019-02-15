from pytensor import *

class MLP(Graph):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__("mlp")

        # make graph
        self.affine1 = self.get_operation('Affine', {'input_size': input_size, 'hidden_size': hidden_size})
        self.sigmoid = self.get_operation('Sigmoid')
        self.affine2 = self.get_operation('Affine', {'input_size': hidden_size, 'hidden_size': output_size})
        self.softmaxloss = self.get_operation('SoftmaxLoss')

    def forward(self, input_tensor):
        affine1_tensor = self.affine1.forward(input_tensor)
        sigmoid_tensor = self.sigmoid.forward(affine1_tensor)
        affine2_tensor = self.affine2.forward(sigmoid_tensor)

        return self.softmaxloss.forward(affine2_tensor)

    def loss(self, target_tensor):
        return self.softmaxloss.loss(target_tensor)
