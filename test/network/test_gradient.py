import numpy as np
import matplotlib.pylab as plt

from network.common.gradient import *

def function_1(x):
    return 0.01*x**2 + 0.1*x

print(numerical_diff(function_1, 10))

def function_2(x):
    return x[0]**2 + x[1]**2

print(numerical_gradient(function_2, np.array([3.0, 4.0])))