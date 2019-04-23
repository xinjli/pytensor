from pytensor.network.tensor import *
import numpy as np
import pickle

def write_parameter(parameter, path):
    """
    save the parameter

    :param parameter: a Parameter instance
    :param path: path to save
    :return:
    """

    np.save(str(path), parameter.tensor_dict)


def read_parameter(path):
    """
    load the parameter from disk

    :param path: path to a npy file
    :return: Parameter instance
    """

    parameter = Parameter()

    # setup parameter
    parameter.tensor_dict = np.load(str(path))[()]

    return parameter


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

    def get_embedding(self, name, shape):
        """
        retrieve a embedding with its name

        :param name: name for embeding
        :param shape: (vocab_size, word_dim)
        :return:
        """

        # get current embedding if it is created already
        if self.embeddings != None:
            return self.embeddings

        # shape should be (vocab_size, word_dim)
        assert len(shape) == 2
        vocab_size, word_dim = shape

        # get tensor for embedding
        embedding_tensor = self.get_tensor(name, shape)

        # initialize the embedding
        self.embeddings = []

        # embedding is implemented as a list of tensors
        # this is for efficient update
        for i in range(vocab_size):

            # create reference to embedding tensor
            value = embedding_tensor.value[i].reshape(1,word_dim)
            grad = embedding_tensor.grad[i].reshape(1, word_dim)

            embedding = Tensor(value=value, grad=grad)
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
