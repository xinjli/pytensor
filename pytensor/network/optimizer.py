# coding: utf-8
import numpy as np
import logging
from pytensor.network.parameter import *

class Optimizer:

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, parameter, lr=0.01):
        self.parameter = parameter
        self.lr = lr

    def step(self):

        for param_name in self.parameter.tensor_dict.keys():
            # update param value
            param = self.parameter.tensor_dict[param_name]

            # update
            if param.trainable:
                param.value -= self.lr * param.grad

        for temp_tensor in self.parameter.temp_tensors:

            # update temp_tensor
            if temp_tensor:
                temp_tensor.value -= self.lr * temp_tensor.grad

    def zero_grad(self):

        # clear all gradients
        self.parameter.clear_grads()

        # clear temp tensors
        self.parameter.clear_temp_tensor()

