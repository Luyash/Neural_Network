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

    # =====================
    # FORWARD PROPAGATION
    # =====================

    # Layer 1:
    A1 = X_train_scaled @ W1 + b1
    output1 = relu(A1)

    # Layer 2:
    A2 = output1 @ W2 +b2
    output2 = relu(A2)

    # Output:
    predicted = output2 @ W3 + b3

    # Loss layer 3:
    loss = np.mean((predicted - y_train) ** 2)
    print(f"The loss of epoch no: {epoch} is --> {loss}")

    d_predicted = (2 * (predicted - y_train)) /len(y_train)

    # =====================
    # BACKPROPAGATION
    # =====================

    # Gradients:
    # Output Layer:
    w3_gradient = output2.T @ d_predicted
    b3_gradient = np.sum(d_predicted, axis=0)

    # Send gradient back to output2:
    doutput2 = d_predicted @ W3.T

    #Layer2
    # Relu
    dA2 = doutput2 * (A2 > 0)
     
    w2_gradient = output1.T @ dA2
    b2_gradient = np.sum(dA2, axis=0)

    # Send gradient backwards to output1
    doutput1 = dA2 @ W2.T

    # Layer 1
    dA1 = doutput1 * (A1 > 0)

    w1_gradient = X_train_scaled.T @ dA1
    b1_gradient = np.sum(dA1, axis=0)

    # =====================
    # UPDATE WEIGHTS
    # =====================

    # Gradient Descent
    W3 = W3 - learning_rate * w3_gradient
    b3 = b3 - learning_rate * b3_gradient

    W2 = W2 - learning_rate * w2_gradient
    b2 = b2 - learning_rate * b2_gradient

    W1 = W1 - learning_rate * w1_gradient
    b1 = b1 - learning_rate * b1_gradient












