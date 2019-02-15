from pytensor import *

class RNNClassifier(Graph):

    def __init__(self, vocab_size, input_size, hidden_size):
        """
        RNN classification

        :param vocab_size:
        :param input_size:
        :param hidden_size:
        """

        super().__init__('RNNClassifier')

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

        # word embedding
        embed_argument = {'vocab_size': self.vocab_size, 'embed_size': self.input_size}
        self.word_embedding = self.get_operation('Embedding', embed_argument)

        # rnn
        rnn_argument = {'input_size': self.input_size, 'hidden_size': self.hidden_size, 'max_num_steps': self.max_num_steps}
        self.rnn = self.get_operation('RNN', rnn_argument)

        # affines
        self.affine = self.get_operation('Affine', {'input_size': self.hidden_size, 'hidden_size': self.output_size}, "Affine")

        # softmax
        self.softmaxLoss = self.get_operation('SoftmaxLoss')

    def forward(self, word_lst):

        # get num steps
        self.num_steps = min(len(word_lst), self.max_num_steps)

        # create embeddings
        self.embedding_tensors = []
        for word_id in word_lst:
            self.embedding_tensors.append(self.word_embedding.forward([LongTensor([word_id])]))

        # run RNN
        self.rnn_tensors = self.rnn.forward(self.embedding_tensors)

        # affine
        self.output_tensor = self.affine.forward(self.rnn_tensors[-1])

        self.softmax_tensor = self.softmaxLoss.forward(self.output_tensor)

        return self.softmax_tensor

    def loss(self, target_id):

        ce_loss = self.softmaxLoss.loss(LongTensor([target_id]))
        return ce_loss


class RNNLM(Graph):

    def __init__(self, vocab_size, input_size, hidden_size):
        super().__init__('RNN')

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

        # word embedding
        embed_argument = {'vocab_size': self.vocab_size, 'embed_size': self.input_size}
        self.word_embedding = self.get_operation('Embedding', embed_argument)

        # rnn
        rnn_argument = {'input_size': self.input_size, 'hidden_size': self.hidden_size, 'max_num_steps': self.max_num_steps}
        self.rnn = self.get_operation('RNN', rnn_argument)

        # affines
        affine_argument = {'input_size': self.hidden_size, 'hidden_size': self.output_size}
        self.affines = [self.get_operation('Affine', affine_argument, "Affine") for i in range(self.max_num_steps)]

        # softmax
        self.softmaxLosses = [self.get_operation('SoftmaxLoss') for i in range(self.max_num_steps)]

    def forward(self, word_lst):

        # get num steps
        self.num_steps = min(len(word_lst), self.max_num_steps)

        # create embeddings
        embedding_tensors = []
        for word_id in word_lst:
            embedding_tensors.append(self.word_embedding.forward([LongTensor([word_id])]))

        # run RNN
        rnn_tensors = self.rnn.forward(embedding_tensors)

        # softmax tensors
        softmax_tensors = []

        for i in range(self.num_steps):
            output_tensor = self.affines[i].forward(rnn_tensors[i])
            softmax_tensor = self.softmaxLosses[i].forward(output_tensor)
            softmax_tensors.append(softmax_tensor)

        return softmax_tensors

    def loss(self, target_ids):

        ce_loss = 0.0

        for i in range(self.num_steps):
            cur_ce_loss = self.softmaxLosses[i].loss(LongTensor([target_ids[i]]))
            ce_loss += cur_ce_loss

        return ce_loss