import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def relu(x):
    return np.where(x<0, 0, x)

def tanh(x):
    return np.tanh(x)
