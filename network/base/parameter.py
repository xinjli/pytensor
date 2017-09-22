from network.base.variable import *
from common.config import *
import numpy as np
import pickle


def load_parameter(filename):
    filepath = config.pyai_path+'corpus/pickle/'+filename+'.pkl'
    return pickle.load(open(filepath, 'rb'))


def dump_parameter(obj, filename):
    filepath = config.pyai_path+'corpus/pickle/'+filename+'.pkl'
    pickle.dump(obj, open(filepath, 'wb'))


class Parameter:
    """
    Parameter
    """

    def __init__(self):
        self.variable_dict = dict()

        self.embeddings = None
        self.temp_variables = []

    def get_variable(self, name, shape):

        if name in self.variable_dict:
            return self.variable_dict[name]
        else:
            # initialize a new variable for it
            value = np.random.standard_normal(shape) / np.sqrt(shape[0])
            variable = Variable(value, name=name)

            # register it
            self.variable_dict[name] = variable

            return variable

    def get_embedding(self, vocab_size, word_dim):

        if self.embeddings != None:
            return self.embeddings

        self.embeddings = []
        for i in range(vocab_size):
            embedding = Variable(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), word_dim))
            self.embeddings.append(embedding)

        return self.embeddings


    def add_temp_variable(self, temp_variable):
        """
        register temporary variable for optimizer to update variable
        this is mainly for word embedding training

        :param temp_variable:
        :return:
        """

        self.temp_variables.append(temp_variable)


    def clear_temp_variable(self):
        """
        clear param embedding list

        :return:
        """
        self.temp_variables = []


    def clear_grads(self):
        """
        clear gradients of all variables

        :return:
        """

        for k, v in self.variable_dict.items():
            v.clear_grad()

        for v in self.temp_variables:
            v.clear_grad()


    def validate_gradient(self, loss_func, name):
        """
        validate numerical gradient with respect to a loss func

        :param loss_func:
        :param name: variable's name which we want to check gradient
        :return:
        """

        x = self.variable_dict[name].value

        h = 1e-4
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = loss_func()

            x[idx] = tmp_val - h
            fxh2 = loss_func()  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val
            it.iternext()

        print("Gradient ", grad)