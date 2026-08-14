import pandas as pd
import numpy as np
from functions import relu


# Take the pandas DataFrame called data and convert it into a NumPy array
data = pd.read_excel("Real Estate/Real estate valuation data set.xlsx")


# this makes array like ndarray datatype of numpy 
array = data.to_numpy()

# Inputs and Targets
X = array[:, 1:7]
y = array[:, 7]

# Splitting the data into 80% training and 20% test
split_index = int(len(X) * 0.8)  # Gives 80% vaneko kati ho vanera 

X_train = X[:split_index] # First 80%
X_test = X[split_index:]  # Remaining 20%

y_train = y[:split_index]  # First 80%
y_test = y[split_index:]   # Remaining 20%

# Calculate mean, SD for Standardization:
train_mean = X_train.mean(axis=0)
train_sd = X_train.std(axis=0)

# Standardize
X_train_scaled = (X_train - train_mean) / train_sd
X_test_scaled = (X_test - train_mean) / train_sd

# Okay now the training stuff below:


# Initializing weights and biases:
W1 = np.random.randn(6, 8)  # 6 rows and 8 columns not actual 6x8
b1 = np.zeros(8)

W2 = np.random.randn(8, 8) # This creates array of numpy with 8 row and 8 col
b2 = np.zeros(8)

W3 = np.random.randn(8, 1)
b3 = np.zeros(1)

learning_rate = 0.01

# The main loop:

for epoch in range (1,2001):
    # Layer 1:
    A1 = X_train_scaled @ W1 + b1
    C1 = relu(A1)

    # Layer 2:
    A2 = C1 @ W2 +b2
    C2 = relu(A2)

    # Output:
    predicted = C2 @ W3 + b3

    # Loss:
    loss = np.mean((predicted - y_train) ** 2)

    # Gradients:
    gradient_weight = np.mean(2 * (predicted - y_train) *x)
    gradient_bias = np.mean(2* (predicted - y_train))






