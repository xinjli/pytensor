from network.base.operation import *
from network.base.optimizer import *
from network.base.parameter import *
from network.base.gradient import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class FNN:

    def __init__(self, input_size, hidden_size, output_size):

        # initialize model parameters
        self.parameter = Parameter()

        # make graph
        self.affine1 = Affine(input_size, hidden_size, self.parameter, "Affine1")
        self.sigmoid = Sigmoid()
        self.affine2 = Affine(hidden_size, output_size, self.parameter, "Affine2")
        self.softmaxLoss = SoftmaxWithLoss()

        # optimizer
        self.optimizer = SGD(self.parameter)


    def forward(self, input_value):
        input_variable = Variable(input_value)
        affine1_variable = self.affine1.forward(input_variable)
        sigmoid_variable = self.sigmoid.forward(affine1_variable)
        affine2_variable = self.affine2.forward(sigmoid_variable)

        return self.softmaxLoss.forward(affine2_variable)

    def loss(self, output_value):
        return self.softmaxLoss.get_loss(output_value)


    def backward(self):
        self.softmaxLoss.backward()
        self.affine2.backward()
        self.sigmoid.backward()
        self.affine1.backward()

    def update(self):
        self.optimizer.update()

    def accuracy(self, input_lst, output_lst):

        acc_cnt = 0.0
        all_cnt = len(input_lst)

        for i in range(len(input_lst)):
            output_variable = self.forward(input_lst[i])
            y = np.argmax(output_variable.value)
            if y == output_lst[i]:
                acc_cnt += 1.0

        print("Accuracy ", acc_cnt / all_cnt)


    def train(self, input_lst, output_lst, val_input_lst, val_output_lst, epoch=200):

        for ii in range(epoch):

            print("Now epoch ", ii)
            all_loss = 0.0

            for i in range(len(input_lst)):
                # forward
                self.forward(input_lst[i])

                # loss
                all_loss += self.loss(output_lst[i])

                # backward
                self.backward()

                # update parameters
                self.update()

            all_loss /= len(input_lst)
            print("Current loss ", all_loss)
            self.accuracy(val_input_lst, val_output_lst)


if __name__ == '__main__':

    digits = load_digits()
    digits.data /= 16.0
    data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target)

    model = FNN(64, 30, 10)

    validate_gradient(model, data_train[0], label_train[0])
    model.train(data_train, label_train, data_test, label_test, 10)
    validate_gradient(model, data_train[0], label_train[0])
