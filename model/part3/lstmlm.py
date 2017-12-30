from network.base.rnn import *
from network.base.embedding import *
from network.base.parameter import *
from network.base.optimizer import *
from network.base.operation import *
from network.base.lstm import *
from network.base.gradient import *
from common.config import *
from data.loader.load_ptb import *
from common.logger import *


class LSTMLM:

    def __init__(self, vocab_size, input_size, hidden_size):

        self.parameter = Parameter()

        # embedding size
        self.vocab_size = vocab_size
        self.word_dim = input_size

        # network size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size

        # num steps
        self.max_num_steps = 100
        self.num_steps = 0

        # optimizer
        self.optimizer = SGD(self.parameter)

        # graph
        self.make_graph()

    def make_graph(self):

        self.word_embedding = WordEmbedding(self.vocab_size, self.input_size, self.parameter)
        self.lstm = LSTM(self.input_size, self.hidden_size, max_num_steps=self.max_num_steps, parameter=self.parameter)
        self.affines = [Affine(self.hidden_size, self.output_size, self.parameter) for i in range(self.max_num_steps)]
        self.softmaxLosses = [SoftmaxWithLoss() for i in range(self.max_num_steps)]


    def forward(self, word_lst):

        # get num steps
        self.num_steps = min(len(word_lst), self.max_num_steps)

        # create embeddings
        embedding_variables = []
        for word_id in word_lst:
            embedding_variables.append(self.word_embedding.forward(word_id))

        # run LSTM
        lstm_variables = self.lstm.forward(embedding_variables)

        # softmax variables
        softmax_variables = []

        for i in range(self.num_steps):
            output_variable = self.affines[i].forward(lstm_variables[i])
            softmax_variable = self.softmaxLosses[i].forward(output_variable)
            softmax_variables.append(softmax_variable)

        return softmax_variables

    def loss(self, target_ids):

        ce_loss = 0.0

        for i in range(self.num_steps):
            cur_ce_loss = self.softmaxLosses[i].get_loss(target_ids[i])
            ce_loss += cur_ce_loss

        return ce_loss

    def backward(self):

        # softmax backward
        for i in range(self.num_steps):
            self.softmaxLosses[i].backward()
            self.affines[i].backward()

        # rnn backward
        self.lstm.backward()


    def train(self, input_lst, output_lst, epoch=100, iteration=10):


        for ii in range(epoch):

            print("Epoch ", ii)

            ce_loss = 0.0
            ce_cnt = 0.0

            it_ce_loss = 0.0
            it_ce_cnt = 0.0

            for i in range(len(input_lst)):

                # extract data set
                input_ids = input_lst[i]
                output_ids = output_lst[i]


                # regular steps
                self.forward(input_ids)

                cur_ce_loss = self.loss(output_ids)
                cur_ce_cnt = self.num_steps

                it_ce_loss += cur_ce_loss
                it_ce_cnt += cur_ce_cnt

                ce_loss += cur_ce_loss
                ce_cnt += cur_ce_cnt

                self.backward()

                self.optimizer.update()

                if (i+1) % iteration == 0:
                    # report iteration
                    print("Current iteration: ", i+1, " ce loss: ", it_ce_loss / it_ce_cnt)

                    # clear ce loss
                    it_ce_loss = 0.0
                    it_ce_cnt = 0.0

            ce_loss /= ce_cnt
            print("Perplexity ", 2.0**ce_loss)


def check_simple_case():

    max_time_steps = 5

    input_lst =  [[np.random.randint(2) for i in range(max_time_steps)] for j in range(5)]
    output_lst = [[np.random.randint(2) for i in range(max_time_steps)] for j in range(5)]

    lstmlm = LSTMLM(2, 2, 2)
    validate_gradient(lstmlm, input_lst[0], output_lst[0])
    lstmlm.train(input_lst, output_lst, 1)
    validate_gradient(lstmlm, input_lst[0], output_lst[0])


def check_ptb():

    sentences, vocab = load_ptb()
    input_lst = []
    output_lst = []

    for sentence in sentences:
        input_ids = sentence[:-1]
        output_ids = sentence[1:]

        input_lst.append(input_ids)
        output_lst.append(output_ids)

    lstmlm = LSTMLM(10000, 100, 100)

    #validate_gradient(lstmlm, input_lst[0], output_lst[0])
    lstmlm.train(input_lst, output_lst, 100, 1000)
    #validate_gradient(lstmlm, input_lst[0], output_lst[0])



if __name__ == '__main__':
    check_ptb()
    #check_simple_case()


