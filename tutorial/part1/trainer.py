from network.base.optimizer import *

class Trainer:

    def __init__(self, model):
        """
        A trainer example

        :param model:
        """

        self.model = model
        self.optimizer = SGD(self.model.parameter)

    def train(self, x_train, y_train, x_test=None, y_test=None, epoch=40, iteration=10000):

        for ii in range(epoch):

            loss = 0.0
            accuracy = 0.0

            it_loss = 0.0

            for i in range(len(x_train)):

                # extract data set
                input_variables = Variable([x_train[i]])
                target_variable = Variable([y_train[i]])

                # dynamic forward
                self.model.forward(input_variables)
                cur_loss = self.model.loss(target_variable)

                it_loss += cur_loss
                loss += cur_loss

                self.model.backward()

                self.optimizer.update()

                if (i+1) % iteration == 0:
                    # report iteration
                    print("=== Epoch: ", ii, " Iteration: ", i+1, " train loss: ", it_loss / iteration, " ===")

                    # clear ce loss
                    it_loss = 0.0

                    self.test(x_test, y_test)

            print("=== Epoch ", ii, " Summary ===")
            self.test(x_test, y_test)


    def test(self, x_test, y_test):

        acc_cnt = 0.0
        all_cnt = len(x_test)

        for i in range(len(x_test)):

            v = Variable([x_test[i]])
            output_variable = self.model.forward(v)

            y = np.argmax(output_variable.value[0])
            if y == y_test[i]:
                acc_cnt += 1.0

        print("test accuracy ", acc_cnt / all_cnt)
