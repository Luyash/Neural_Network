import numpy as np
import time

# We will now try to predict the equation y = 3x + 2z + 1
# This has 2 unknown variables

# Initializing values:
x1 = np.array([
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9
])

x2 = np.array([
    -5, -4, -3, -2,
    -1, 0, 1, 2, 3, 4
])

y = np.array([
    -5, 0, 5, 10, 15,
    20, 25, 30, 35, 40
])

learning_rate = 0.01

# Initial Weight and Bias:
w1 = -0.6
w2 = 6.67
b = 3.1

# Main loop
for epoch in range(1, 801):
    predicted = w1 * x1 + w2 * x2 + b

    # Predicted is an array of predicted values and y is the actual array of values soo loss function
    loss = np.mean((predicted - y) ** 2)
    time.sleep(0.01)
    print(f"The loss of epoch number: {epoch} is --> {loss}")

    # Calculate Gradients:
    w1_gradient = np.mean(2 * (predicted - y) * x1)
    w2_gradient = np.mean(2 * (predicted - y) * x2)
    b_gradient = np.mean(2 * (predicted - y))

    # Gradient_descent:
    w1 = w1 - learning_rate * w1_gradient
    w2 = w2 - learning_rate * w2_gradient
    b = b - learning_rate * b_gradient

    print("="*70)
    print(f"Predicted equation is --> y = {w1}x1 + {w2}x2 + {b}")
    print("="*70)

print()
print("Prediction is completed")
print("="*70)
print()
print(f"Final predicted equation is --> y = {round(w1)}x1 + {round(w2)}x2 + {round(b)}")
print()
print("="*70)
print()
print("="*70)
a = int(input("Enter the value of x1 --> "))
c = int(input("Enter the value of x2 --> "))
print("="*70)
print()
result = w1 * a + w2 * c + b
print(f"The value of y is --> {result} ")
print("="*70)




     
    




