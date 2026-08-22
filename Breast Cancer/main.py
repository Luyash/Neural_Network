import zipfile
import numpy as np
from functions import *

with zipfile.ZipFile("breast+cancer+wisconsin+diagnostic.zip", "r") as zip_file:
    zip_file.extract("wdbc.data")

data = np.genfromtxt(
    "wdbc.data",
    delimiter=",",
    dtype=None,
    encoding="utf-8"
)

# Getting data Inputs and Output:
X = data[:, 2:32]   # columns 2 through 31 → 30 inputs

y = data[:, 1:2]    # column 1 → diagnosis

# Adding randomizer for the input data like biccha biccha bata 80% and 20%
indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]

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


# Now the actual fun part:

# Initializing weights and biases:
W1 = np.random.randn(30, 16) * np.sqrt(2 / 30)  # Random values but around +-1
b1 = np.zeros(16)

W2 = np.random.randn(16, 8) * np.sqrt(2 / 16) # This kind of initialization is called He Initialization This is good for Relu activation function
b2 = np.zeros(8)

W3 = np.random.randn(8, 1) # No He initialization here because aaba Sigmoid lai dine not ReLu; rakhda nee kei huna chhai hudaina huna chhai 
b3 = np.zeros(1)

learning_rate = 0.001
batch_size = 32

# The main loop

for epoch in range (1,20000,1):

    # Forward Propagation:

    # Layer 1
    intermediate_output1 = X_train_scaled @ W1 + b1
    output_1 = relu(intermediate_output1)

    # Layer 2
    intermediate_output2 = output_1 @ W2 + b2
    output_2 = relu(intermediate_output2)

    #Layer 3 
    intermediate_output3 = output_2 @ W3 + b3
    predicted = sigmoid(intermediate_output3)

    # Loss for the predicted values:
    loss = BCE(y_train , predicted)

    # =====================
    # BACKPROPAGATION
    # =====================

    d_predicted = derivative_BCE(y_train, predicted)  # dloss/ dpredicted

    d_sigmoid3 = derivative_sigmoid()

    dW3 = intermediate_output3.T @ d_predicted
    db3 = np.sum(d_predicted , axis = 0)

    d_output2 = 
    











