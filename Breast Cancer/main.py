import zipfile
import numpy as np

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

# Main Training:


