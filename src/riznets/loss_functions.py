import numpy as np

def mse(y, a):
    return 0.5 * np.sum(np.power((a - y), 2))

def mse_d(y, a):
    return np.diag(a - y)

tiny_const = 0.0000000005

def ce(y, a):
    return -np.sum(y * np.log(a + tiny_const) + (1 - y) * np.log(1 - a + tiny_const))

def ce_d(y, a):
    return (a - y) / ((1 - a + tiny_const) * (a + tiny_const))

loss = {
    "mse" : mse,
    "ce" : ce
}

loss_d = {
    "mse" : mse_d,
    "ce" : ce_d
}
