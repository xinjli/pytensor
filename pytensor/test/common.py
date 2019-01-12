import unittest
import numpy as np

def gradient_generator(model, input_variables, target_variable):
    """
    generate diference for testing purpose

    :param model: the model we want to validate
    :param input_variables: input variables for gradient check
    :param target_variable: target variable for gradient check
    :return:
    """

    # clear gradient before validation
    model.graph.parameter.clear_grads()

    # run normal procedures to compute automatic gradient
    model.forward(input_variables)
    model.loss(target_variable)
    model.graph.backward()

    # variables we need to check
    variable_dict = model.graph.parameter.variable_dict

    for var_name, var in variable_dict.items():

        h = 1e-4
        v = var.value

        numerical_grad = np.zeros_like(v)

        # compute numerical gradient of this variable
        it = np.nditer(v, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = v[idx]

            # f(x+h)
            v[idx] = float(tmp_val) + h
            model.forward(input_variables)
            loss_1 = model.loss(target_variable)

            # f(x-h)
            v[idx] = tmp_val - h
            model.forward(input_variables)
            loss_2 = model.loss(target_variable)

            numerical_grad[idx] = (loss_1 - loss_2) / (2 * h)

            v[idx] = tmp_val
            it.iternext()

            # clear ops
            model.graph.clear()

        # compare numerical grad with auto grad
        diff = np.sum(var.grad - numerical_grad)

        yield (var_name, numerical_grad, var.grad)