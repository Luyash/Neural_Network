import numpy as np
import math

def relu(x):   # Relu ---> negative and 0 return 0 natra return input number
    if x <= 0:
        return 0
    else:
        return x

def sigmoid(x):    # Sigmoid function --> y = 1/ ( 1+ e^(-x) )
    y = 1 + math.exp(-x)
    y = 1 / y
    return y