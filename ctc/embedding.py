from pytensor.network.variable import *
from pytensor.network.operation import *

class LookupEmbedding:

    def __init__(self, word_embedding, axis=0):
        """
        :param word_embedding: Word embedding
        :param axis: axis how to concatenate words when input is a list
                     vstack when axis = 0
                     hstack when axis = 1

        """

        self.batch = True
        self.word_embedding = word_embedding

        self.axis = axis
        if self.axis == 0:
            self.stack = VStack()
        else:
            self.stack = HStack()

    def forward(self, x, trainable=True):
        """
        :param x: word index or its list
        :return: variable
        """

        if isinstance(x, int):
            # expand x into batch
            x = [x]

        # input is a list of id
        input_variables = [self.word_embedding.forward(word_id, trainable=trainable) for word_id in x]
        output_variable = self.stack.forward(input_variables)

        return output_variable

    def backward(self):

        if self.batch:
            self.stack.backward()


class WordEmbedding:

    def __init__(self, vocab_size, word_dim, parameter):

        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.parameter = parameter
        self.embedding_variables = self.parameter.get_embedding(vocab_size, word_dim)

    def forward(self, word_id, trainable=True):
        """
        :param word_id: word index

        :return: variable

        """
        assert (word_id < self.vocab_size)
        output_variable = self.embedding_variables[word_id]

        if trainable:
            self.parameter.add_temp_variable(output_variable)

        return output_variable


    def backword(self):
        return