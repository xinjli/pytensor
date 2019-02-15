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

        self.embedding_tensors = self.graph.parameter.get_embedding(self.vocab_size, self.embed_dim)

    def forward(self, input_tensors):
        """
        get the embedding

        :param input_tensors: input tensor is a LongTensor containing word ids
        :return: embedding
        """
        self.register(input_tensors)

        # embedding only takes 1 input tensor
        assert(len(input_tensors) == 1)

        word_id = input_tensors[0].value[0]

        assert (word_id < self.vocab_size)
        output_tensor = self.embedding_tensors[word_id]

        if self.trainable:
            self.graph.parameter.add_temp_tensor(output_tensor)

        return output_tensor

    def backward(self):
        return