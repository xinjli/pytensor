from pytensor.tutorial.part2.trainer import *
from pytensor.data.digit_dataset import *


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



def linear_train():
    data_train, data_test, label_train, label_test = digit_dataset()
    model = Linear(64, 10)

    trainer = Trainer(model)
    trainer.train(data_train, label_train, data_test, label_test, 40)


if __name__ == '__main__':
    linear_train()