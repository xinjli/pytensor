from network.temp.ctc import *

if __name__ == "__main__":

    ctc = SoftmaxCTCLoss()

    vars = []

    vars.append(Variable([1,2,3,4,5]))
    vars.append(Variable([2,3,4,5,6]))
    vars.append(Variable([3,4,5,6,7]))
    vars.append(Variable([4,5,6,7,8]))
    vars.append(Variable([5,6,7,8,9]))

    outputs = ctc.forward(vars)

    for output in outputs:
        print(output.value)


    print(ctc.loss([1, 1]))
    for node_line in ctc.lattice:
        print(' '.join([str(node.ln_alpha) for node in node_line]))


    ctc.backward()

    for i in range(len(vars)):
        print(vars[i].grad)