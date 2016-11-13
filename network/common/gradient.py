import numpy as np

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h))/2*h

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):

    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x