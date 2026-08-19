import numpy as np
import math


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    y = 1 + math.exp(-x)
    y = 1 / y
    return y