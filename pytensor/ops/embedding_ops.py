from pytensor.network.operation import *

class Embedding(Operation):

    def __init__(self, name='embedding', argument=None, graph=None):
        """
        :param name:
        :param argument:
        - vocab_size: vocabulary size
        - embed_size: embedding size
        - trainable: whether the embedding can be trained

        :param graph:
        """

        super(Embedding, self).__init__(name, argument, graph)

        self.vocab_size = argument['vocab_size']
        self.embed_dim = argument['embed_size']

        if 'trainable' in argument:
            self.trainable = argument['trainable']
        else:
            self.trainable = True

        self.embedding_variables = self.graph.parameter.get_embedding(self.vocab_size, self.embed_dim)

    def forward(self, input_variables):
        """
        get the embedding

        :param input_variables: input variable is a LongVariable containing word ids
        :return: embedding
        """
        self.register(input_variables)

        # embedding only takes 1 input variable
        assert(len(input_variables) == 1)

        word_id = input_variables[0].value[0]

        assert (word_id < self.vocab_size)
        output_variable = self.embedding_variables[word_id]

        if self.trainable:
            self.graph.parameter.add_temp_variable(output_variable)

        return output_variable

    def backward(self):
        return