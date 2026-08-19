import numpy as np
import math


# Relu ---> negative and 0 return 0 natra return input number
def relu(x):   
    if x <= 0:
        return 0
    else:
        return x


# Sigmoid function --> y = 1/ ( 1+ e^(-x) )
def sigmoid(x):    
    y = 1 + math.exp(-x)
    y = 1 / y
    return y


# Binary Cross Entropy Loss Function (BCE):
# Used when outputs are 2 loke 0 and 1 or yes and no type
def BCE(actual , predicted):
    loss = -(actual * math.log(predicted) + (1 - actual) * math.log(1 - predicted) )
    return loss