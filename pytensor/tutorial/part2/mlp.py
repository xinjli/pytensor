from pytensor.tutorial.part2.trainer import *
from pytensor.data.digit_dataset import *


class MLP(Graph):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__("mlp")

        # make graph
        self.affine1 = self.get_operation('Affine', {'input_size': input_size, 'hidden_size': hidden_size})
        self.sigmoid = self.get_operation('Sigmoid')
        self.affine2 = self.get_operation('Affine', {'input_size': hidden_size, 'hidden_size': output_size})
        self.softmaxloss = self.get_operation('SoftmaxLoss')

    def forward(self, input_variable):
        affine1_variable = self.affine1.forward(input_variable)
        sigmoid_variable = self.sigmoid.forward(affine1_variable)
        affine2_variable = self.affine2.forward(sigmoid_variable)

        return self.softmaxloss.forward(affine2_variable)

    def loss(self, target_variable):
        return self.softmaxloss.loss(target_variable)


def mlp_train():

    data_train, data_test, label_train, label_test = digit_dataset()
    model = MLP(64, 30, 10)

    trainer = Trainer(model)
    trainer.train(data_train, label_train, data_test, label_test, 40)


if __name__ == '__main__':
    mlp_train()