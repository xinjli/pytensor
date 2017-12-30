import numpy as np

def numerical_gradient_check(model, input_variables, target_variable):
    """
    validate numerical gradient with respect to model

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
        print("Now checking ", var)

        h = 1e-4
        numerical_grad = np.zeros_like(var)

        # compute numerical gradient of this variable
        it = np.nditer(var, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = var[idx]

            # f(x+h)
            var[idx] = float(tmp_val) + h
            model.forward(input_variables)
            loss_1 = model.loss(target_variable)

            # f(x-h)
            var[idx] = tmp_val - h
            model.forward(input_variables)
            loss_2 = model.loss(target_variable)

            numerical_grad[idx] = (loss_1 - loss_2) / (2 * h)

            var[idx] = tmp_val
            it.iternext()

        # compare numerical grad with auto grad
        diff = np.sum(var.grad - numerical_grad)

        if diff < 0.001:
            print(var_name, " match")
        else:
            print(var_name, " NOT MATCH !")
            print("Auto grad\n", var.grad)
            print("Numerical grad\n", numerical_grad)