# coding: utf-8
import numpy as np
import logging
from network.base.parameter import *
from common.logger import *


class SGD:
    def __init__(self, parameter, lr=0.001):
        self.parameter = parameter
        self.lr = lr

    def update(self):

        for param_name in self.parameter.variable_dict.keys():
            # update param value
            param = self.parameter.variable_dict[param_name]

            # report logging
            if debug:
                info = "SGD logging: parameter: " + str(param_name) + " value " + str(param.value)+" grad "+str(param.grad)
                logging.info(info)

            # update
            if param.trainable:
                param.value -= self.lr * param.grad

        for temp_variable in self.parameter.temp_variables:

            # report logging
            if debug:
                info = "SGD logging: parameter: " + str(temp_variable.name) + " value " + str(temp_variable.value)+" grad "+str(temp_variable.grad)
                logging.info(info)

            # update temp_variable
            if temp_variable:
                temp_variable.value -= self.lr * temp_variable.grad


        # clear all gradients
        self.parameter.clear_grads()

        # clear temp variables
        self.parameter.clear_temp_variable()