import unittest
import numpy as np

def gradient_generator(model, input_tensors, target_tensor):
    """
    generate diference for testing purpose

    :param model: the model we want to validate
    :param input_tensors: input tensors for gradient check
    :param target_tensor: target tensor for gradient check
    :return:
    """

    # clear gradient before validation
    model.parameter.clear_grads()

    # run normal procedures to compute automatic gradient
    model.forward(input_tensors)
    model.loss(target_tensor)
    model.backward()

    # tensors we need to check
    tensor_dict = model.parameter.tensor_dict

    for var_name, var in tensor_dict.items():

        h = 1e-4
        v = var.value

        numerical_grad = np.zeros_like(v)

        # compute numerical gradient of this tensor
        it = np.nditer(v, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = v[idx]

            # f(x+h)
            v[idx] = float(tmp_val) + h
            model.forward(input_tensors)
            loss_1 = model.loss(target_tensor)

            # f(x-h)
            v[idx] = tmp_val - h
            model.forward(input_tensors)
            loss_2 = model.loss(target_tensor)

            numerical_grad[idx] = (loss_1 - loss_2) / (2 * h)

            v[idx] = tmp_val
            it.iternext()

            # clear ops
            model.clear()

        # compare numerical grad with auto grad
        diff = np.sum(var.grad - numerical_grad)

        yield (var_name, numerical_grad, var.grad)