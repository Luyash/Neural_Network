import numpy as np
import time

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
learning_rate = 0.01

#The main part:
for epoch in range(1,801):
    prediction = w * x + b

    #loss:
    loss = np.mean((prediction - y) ** 2)
    print(f"The loss for epoch: {epoch} is {loss}")
    time.sleep(0.01)

    gradient_weight = np.mean(2 * (prediction - y) *x)
    gradient_bias = np.mean(2* (prediction - y))

    #Gradient Descent:
    w = w - learning_rate * gradient_weight
    b = b - learning_rate * gradient_bias

print("="*70)
print("weight:", w)
print("bias:", b)
print("="*70)
print(f"The predicted function is y = {w:.4f}x + {b:.4f}")
a = int(input("Enter the value of x --> "))
print("="*70)
ans = w*a + b
print(f"The predicted value of y using y = {int(w)}x + {int(b)} is y = {ans}")



