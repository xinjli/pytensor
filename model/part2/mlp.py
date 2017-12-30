from network.base.graph import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from network.base.trainer import *
from network.base.gradient import *


def accuracy(input_lst, output_lst):
    acc_cnt = 0.0
    all_cnt = len(input_lst)

    for i in range(len(input_lst)):
        output_variable = self.forward(input_lst[i])
        y = np.argmax(output_variable.value)
        if y == output_lst[i]:
            acc_cnt += 1.0

    print("Accuracy ", acc_cnt / all_cnt)


def generate_dataset():

    digits = load_digits()
    digits.data /= 16.0
    data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target)





class MLP:

    def __init__(self, input_size, hidden_size, output_size):

        self.graph = Graph("MLP")

        # make graph
        self.affine1 = self.graph.get_operation('Affine', 'affine/1', [input_size, hidden_size])
        self.sigmoid = self.graph.get_operation('Sigmoid')
        self.affine2 = self.graph.get_operation('Affine', 'affine/2', [hidden_size, output_size])
        self.softmaxloss = self.graph.get_operation('SoftmaxLoss')


    def forward(self, input_value):
        input_variable = Variable(input_value)
        affine1_variable = self.affine1.forward([input_variable])
        sigmoid_variable = self.sigmoid.forward([affine1_variable])
        affine2_variable = self.affine2.forward([sigmoid_variable])

        return self.softmaxloss.forward([affine2_variable])

    def loss(self, target_value):
        target_variable = Variable([target_value])
        return self.softmaxloss.loss(target_variable)


if __name__ == '__main__':

    digits = load_digits()
    digits.data /= 16.0
    data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target)

    model = MLP(64, 30, 10)
    numerical_gradient_check(model, data_train[0], label_train[0])
