import pandas as pd
import numpy as np


# Take the pandas DataFrame called data and convert it into a NumPy array
data = pd.read_excel("Real Estate/Real estate valuation data set.xlsx")


# this makes array like ndarray datatype of numpy 
array = data.to_numpy()

# Get all data:
transaction_date = array[:, 1]
house_age        = array[:, 2]
mrt_distance     = array[:, 3]
stores           = array[:, 4]
latitude         = array[:, 5]
longitude        = array[:, 6]
price            = array[:, 7]


# Splitting data to 80% train and 20% test:
X_train = X[:80%]
X_test = X[80%:]

# Calculate mean, SD for Standardization:
train_mean = X_train.mean(axis=0)
train_sd = X_train.std(axis=0)

# Standardize
X_train_scaled = (X_train - train_mean) / train_sd
X_test_scaled = (X_test - train_mean) / train_sd


