from pytensor.network.base.graph import *

class Trainer:

    def __init__(self, model):
        """
        GraphTrainer takes advantages of graph structure
        :param model:
        """

        self.model = model

    def train(self, x_train, y_train, x_test=None, y_test=None, epoch=40, iteration=1000):

        for ii in range(epoch):

            loss = 0.0

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

                # automatic differentiation
                self.model.graph.backward()

                # optimization
                self.model.graph.optimizer.update()

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