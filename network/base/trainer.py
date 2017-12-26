from network.base.optimizer import *

class Trainer:

    def __init__(self, model):

        self.model = model
        self.optimizer = SGD(self.model.parameter)

    def train(self, x_train, y_train, x_test=None, y_test=None, epoch=40, iteration=10000):

        for ii in range(epoch):

            loss = 0.0
            accuracy = 0.0

            it_loss = 0.0

            for i in range(len(x_train)):

                # extract data set
                x = x_train[i]
                y = y_train[i]

                # regular steps
                self.model.forward(x)

                cur_loss = self.model.loss(y)

                it_loss += cur_loss
                loss += cur_loss

                self.model.backward()

                self.optimizer.update()

                if (i+1) % iteration == 0:
                    # report iteration
                    print("=== Epoch: ", ii, " Iteration: ", i+1, " train loss: ", it_loss / iteration, " ===")

                    # clear ce loss
                    it_loss = 0.0

            print("=== Epoch ", ii, " Summary ===")
            print("train loss: ", loss/len(x_train))


            test_loss = 0.0
            for i in range(len(x_test)):

                x = x_test[i]
                y = y_test[i]

                self.model.forward(x)
                test_loss += self.model.loss(y)
            print("test  loss: ", test_loss/len(x_test))