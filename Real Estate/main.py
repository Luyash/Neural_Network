import pandas as pd
import numpy as np


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




