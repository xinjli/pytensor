from pytensor.network.tensor import *
import numpy as np
import pickle


class Parameter:
    """
    Parameter is a structure to manage all trainable tensors in the graph.

    Each trainable tensor should be initialized using Parameter
    """

    def __init__(self):

        # a dictionary mapping names to tensors
        self.tensor_dict = dict()

        # embedding for partial updates
        self.embeddings = None
        self.temp_tensors = []

    def get_tensor(self, name, shape):
        """
        retrieve a tensor with its name

        :param name: name of the tensor
        :param shape: desired shape
        :return:
        """

        if name in self.tensor_dict:
            # if the tensor exists in the dictionary,
            # retrieve it directly
            return self.tensor_dict[name]
        else:
            # if not created yet, initialize a new tensor for it
            value = np.random.standard_normal(shape) / np.sqrt(shape[0])
            tensor = Tensor(value, name=name)

            # register the tensor
            self.tensor_dict[name] = tensor

            return tensor

    def get_embedding(self, vocab_size, word_dim):

        # get current embedding if it is created already
        if self.embeddings != None:
            return self.embeddings

        # initialize the embedding
        self.embeddings = []

        # embedding is implemented as a list of tensors
        # this is for efficient update
        for i in range(vocab_size):
            embedding = Tensor([np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), word_dim)])
            self.embeddings.append(embedding)

        return self.embeddings


    def add_temp_tensor(self, temp_tensor):
        """
        register temporary tensor for optimizer to update tensor
        this is mainly for word embedding training

        :param temp_tensor: a trainable tensor (usually a tensor in the embedding)
        :return:
        """

        self.temp_tensors.append(temp_tensor)


    def clear_temp_tensor(self):
        """
        clear temporary tensor

        :return:
        """
        self.temp_tensors = []


    def clear_grads(self):
        """
        clear gradients of all tensors

        :return:
        """

        for k, v in self.tensor_dict.items():
            v.clear_grad()

        for v in self.temp_tensors:
            v.clear_grad()
