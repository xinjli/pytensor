from network.base.model import *
from network.ops.math_ops import *
from network.ops.loss_ops import *
from network.base.trainer import *


def generate_dataset(num):
    """
    Generate a list of dataset for training

    y = 2*x_1 + 3*x_2 + noise

    :param num: number of dataset
    :return: x, y
    """

    x = []
    y = []

    for i in range(num):

        new_x = np.array([[np.random.uniform(), np.random.uniform()]])
        new_y = np.array([[new_x[0][0]*2 + new_x[0][1]*3 + np.random.normal(0, scale=0.1)]])
        x.append(Variable(new_x))
        y.append(Variable(new_y))

    return x, y

class LinearModel(Model):

    def __init__(self, input_size, output_size):
        """
        a simple linear model: y = w*x

        :param input_size:
        :param output_size:
        """

        # initialize size
        self.input_size = input_size
        self.output_size = output_size

        # initialize parameters
        self.parameter = Parameter()
        self.W = self.parameter.get_variable('weight', [self.input_size, self.output_size])

        # ops and loss
        self.matmul = Matmul()
        self.loss_ops = SquareErrorLoss()

    def forward(self, input_variable):
        output_variable = self.matmul.forward([input_variable, self.W])
        self.loss_ops.forward(output_variable)

        return output_variable

    def loss(self, target_variable):
        return self.loss_ops.loss(target_variable)

    def backward(self):
        self.loss_ops.backward()


if __name__ == '__main__':

    x_train, y_train = generate_dataset(1000)
    x_test, y_test = generate_dataset(100)

    model = LinearModel(2, 1)
    trainer = Trainer(model)
    trainer.train(x_train, y_train, x_test, y_test)