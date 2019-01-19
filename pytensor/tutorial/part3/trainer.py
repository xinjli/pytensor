from pytensor.network.graph import *
from pytensor.network.gradient import *


class Trainer:

    def __init__(self, model):
        """
        A trainer example using graph for autodiff
        :param model:
        """

        self.model = model

    def train(self, x_train, y_train, x_test=None, y_test=None, epoch=40, iteration=10):

        for ii in range(epoch):

            self.model.train()

            loss = 0.0
            it_loss = 0.0
            word_cnt = 0
            it_word_cnt = 0

            for i in range(len(x_train)):

                # extract data set
                input_variables = x_train[i]
                target_variable = y_train[i]

                word_cnt += len(x_train[i])
                it_word_cnt += len(x_train[i])

                # dynamic forward
                self.model.forward(input_variables)
                cur_loss = self.model.loss(target_variable)

                it_loss += cur_loss
                loss += cur_loss

                # automatic differentiation
                self.model.backward()

                # optimization
                self.model.optimize()

                if (i+1) % iteration == 0:
                    # report iteration
                    print("\repoch {} iteration ({}/{}) : loss {} ".format(ii,i+1,len(x_train), it_loss/it_word_cnt), end='')

                    # clear ce loss
                    it_loss = 0.0
                    it_word_cnt = 0

            train_ppl = 2**(loss / word_cnt)
            print("epoch {}:  train_ppl ".format(ii, train_ppl))

    def test(self, x_test, y_test):

        self.model.eval()

        acc_cnt = 0.0
        all_cnt = len(x_test)

        for i in range(len(x_test)):

            v = Variable([x_test[i]])
            output_variable = self.model.forward(v)

            y = np.argmax(output_variable.value[0])
            if y == y_test[i]:
                acc_cnt += 1.0

        print("test accuracy ", acc_cnt / all_cnt)
