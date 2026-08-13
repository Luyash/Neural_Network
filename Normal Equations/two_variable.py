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
    -1, 0, 1, 2, 3
])

y = np.array([
    -5, 0, 5, 10, 15,
    20, 25, 30, 35, 40
])

learning_rate = 0.01

# Main loop
for echos in range(801):
    