import numpy as np

# The equation is y = 3x + 5
# Lets try predict this using simplest form of NN


#x values:
x = np.array([
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
])

#y values:
y = np.array([
    5,
    8,
    11,
    14,
    17,
    20,
    23,
    26,
    29,
    32
])


# Initial parameters:
w = 5     #weight
b = 0.2   #bias

#The main part:
prediction = w * x + b

#loss:
loss = np.mean(prediction - y) ** 2
print(f"The loss for this iteration is {loss}")

gradient_weight = np.mean(2 * (prediction - y) *x)
gradient_bias = np.mean(2* (prediction - y))

#Gradient Descent:
w = w - 0.1 * gradient_weight
b = b - 0.1 * gradient_bias



