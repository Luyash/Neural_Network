import numpy as np
import math


def relu(x):
    if x <= 0:
        return 0
    else:
        return x

def sigmoid(x):
    y = 1 + math.exp(-x)
    y = 1 / y
    return y