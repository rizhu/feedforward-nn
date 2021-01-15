import numpy as np

def identity(x):
    return x

def identity_d(x):
    return np.eye(x.shape[0])
    
def relu(x):
    return np.maximum(x, 0)

def relu_d(x):
    return 1 * (x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    s = np.exp(x - np.max(x))
    return s / np.sum(s)

def softmax_d(x):
    s = softmax(x)
    return s * (1 - s)

activation = {
    "identity" : identity,
    "i" : identity,
    "relu" : relu,
    "sigmoid" : sigmoid,
    "softmax" : softmax
}

activation_d = {
    "identity" : identity_d,
    "i" : identity_d,
    "relu" : relu_d,
    "sigmoid" : sigmoid_d,
    "softmax" : softmax_d
}
