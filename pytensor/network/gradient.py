import numpy as np

def numerical_gradient_check(model, input_tensors, target_tensor):
    """
    validate numerical gradient with respect to model

    :param model: the model we want to validate
    :param input_tensors: input tensors for gradient check
    :param target_tensor: target tensor for gradient check
    :return:
    """

    # clear gradient before validation
    model.graph.parameter.clear_grads()

    # run normal procedures to compute automatic gradient
    model.forward(input_tensors)
    model.loss(target_tensor)
    model.graph.backward()

    # tensors we need to check
    tensor_dict = model.graph.parameter.tensor_dict

    for var_name, var in tensor_dict.items():
        print("Now checking ", var)

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
            model.graph.clear()

        # compare numerical grad with auto grad
        diff = np.sum(var.grad - numerical_grad)

        if diff < 0.001:
            print(var_name, " match")
        else:
            print(var_name, " NOT MATCH !")
            print("Auto grad\n", var.grad)
            print("Numerical grad\n", numerical_grad)