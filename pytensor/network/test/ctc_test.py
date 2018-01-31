from network.temp.ctc import *
from network.base.ctc_decoder import *
from network.base.variable import *
from network.base.gradient import *

def check_simple_ctc_case():

    # test simple case
    # input_variables is [1.0, 2.0, 3.0] * 3
    # and output is [1]
    #
    # when gradient get updated, input_variables should be something like
    # [0, 1, 0] <- match for first label

    input_variables = []
    for i in range(3):
        in_val = np.array([1.0, 2.0, 3.0])
        input_variable = Variable(in_val)
        input_variables.append(input_variable)

    labels = [1]

    ops = SoftmaxCTCLoss()

    for i in range(200):

        ops.forward(input_variables)
        loss_score = ops.loss(labels)
        ops.backward()

        print("Epoch ", i, "Loss score: ", loss_score)

        for j in range(3):
            input_variable.value -= input_variable.grad * 0.01
            input_variable.clear_grad()

            print("Current input ", j, " : ", input_variable.value)


def check_ctc_gradient():
    # test simple case
    # input_variables is [1.0, 2.0, 3.0] * 3
    # and output is [1]
    #
    # when gradient get updated, input_variables should be something like
    # [0, 1, 0] <- match for first label

    input_variables = []
    for i in range(3):
        #in_val = np.array([1.0, 2.0, 3.0])
        in_val = np.random.random(3)

        input_variable = Variable(in_val)
        input_variables.append(input_variable)

    labels = [1]

    ops = SoftmaxCTCLoss()

    ops.forward(input_variables)
    loss_score = ops.loss(labels)
    ops.backward()

    for i in range(3):
        validate_variable_gradient(ops, input_variables, labels, input_variables[i], "Var "+str(i))


def check_complex_ctc_gradient():

    num_frame = 100
    num_class = 50
    num_label = 10

    input_variables = []
    for i in range(num_frame):
        in_val = np.random.random(num_class)

        input_variable = Variable(in_val)
        input_variables.append(input_variable)

    labels = np.random.randint(0, num_class, num_label)

    ops = SoftmaxCTCLoss()

    ops.forward(input_variables)
    loss_score = ops.loss(labels)
    ops.backward()

    for i in range(num_frame):
        validate_variable_gradient(ops, input_variables, labels, input_variables[i], "Var "+str(i))


if __name__ == '__main__':
    print("Thank you")

    #check_simple_ctc_case()
    #check_ctc_gradient()
    #check_complex_ctc_gradient()