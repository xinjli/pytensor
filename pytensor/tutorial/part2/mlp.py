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

    def forward(self, input_tensor):
        affine1_tensor = self.affine1.forward(input_tensor)
        sigmoid_tensor = self.sigmoid.forward(affine1_tensor)
        affine2_tensor = self.affine2.forward(sigmoid_tensor)

        return self.softmaxloss.forward(affine2_tensor)

    def loss(self, target_tensor):
        return self.softmaxloss.loss(target_tensor)


def mlp_train():

    data_train, data_test, label_train, label_test = digit_dataset()
    model = MLP(64, 30, 10)

    trainer = Trainer(model)
    trainer.train(data_train, label_train, data_test, label_test, 40)


if __name__ == '__main__':
    mlp_train()