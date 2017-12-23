# coding: utf-8
import numpy as np
import logging
from network.base.parameter import *


class SGD:
    def __init__(self, parameter, lr=0.001):
        self.parameter = parameter
        self.lr = lr

    def update(self):

        for param_name in self.parameter.variable_dict.keys():
            # update param value
            param = self.parameter.variable_dict[param_name]

            # update
            if param.trainable:
                param.value -= self.lr * param.grad

        for temp_variable in self.parameter.temp_variables:

            # update temp_variable
            if temp_variable:
                temp_variable.value -= self.lr * temp_variable.grad

        # clear all gradients
        self.parameter.clear_grads()

        # clear temp variables
        self.parameter.clear_temp_variable()