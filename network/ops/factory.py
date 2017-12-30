from network.ops.math_ops import *
from network.ops.loss_ops import *
from network.ops.array_ops import *

def create_operation(ops_type, ops_name=None, ops_argument=None, graph=None):

    ops = None

    if ops_type == 'Affine':
        if ops_name is None:
            ops_name = 'affine'

        ops = Affine(ops_name, ops_argument, graph)

    elif ops_type == 'Sigmoid':
        if ops_name is None:
            ops_name = 'sigmoid'

        ops = Sigmoid(ops_name, ops_argument, graph)

    elif ops_type=='SoftmaxLoss':
        if ops_name is None:
            ops_name = 'softmaxloss'

        ops = SoftmaxLoss(ops_name, ops_argument, graph)

    else:
        print(ops, " is not implemented")
        exit()

    return ops
