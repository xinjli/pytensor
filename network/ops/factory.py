from network.ops.math_ops import *
from network.ops.loss_ops import *
from network.ops.array_ops import *

def create_operation(ops_type, ops_name=None, ops_argument=None, graph=None):

    cls = globals()[ops_type]
    return cls(ops_name, ops_argument, graph)