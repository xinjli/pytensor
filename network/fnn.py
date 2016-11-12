import numpy as np
from network.math import *

class FeedforwardNeuralNetwork:

    """
    Normal Feedforward neural network

    """

    def __init__(self, dimensions):
        """
        initialize the network

        :param dimensions: dimensions should be an array of dimension count for each layers
         for instance, to specify a MNIST 3 layer network specify (784, 100, 10)

        """

        self.parameter = {}
