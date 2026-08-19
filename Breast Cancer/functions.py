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
def BCE(actual, predicted):

    total_loss = 0

    for i in range(len(actual)):
        loss = -(actual[i] * math.log(predicted[i]) +(1 - actual[i]) * math.log(1 - predicted[i]))

        total_loss += loss

    return total_loss / len(actual) # Average returning

def derivative_sigmoid(x):
    derivative = sigmoid(x) * (1 - sigmoid(x) )
    return derivative


def derivative_BCE(actual, predicted):
    d_predicted = -(actual / predicted) + ((1 - actual) / (1 - predicted))
    return d_predicted

