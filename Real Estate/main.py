import pandas as pd
import numpy as np
from functions import relu


# Take the pandas DataFrame called data and convert it into a NumPy array
data = pd.read_excel("Real Estate/Real estate valuation data set.xlsx")


# this makes array like ndarray datatype of numpy 
array = data.to_numpy()

# Inputs and Targets
X = array[:, 1:7]
y = array[:, 7:8]

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

# Okay now the training stuff below:


# Initializing weights and biases:
W1 = np.random.randn(6, 8) * np.sqrt(2 / 6)  # Random values but around +-1
b1 = np.zeros(8)

W2 = np.random.randn(8, 8) * np.sqrt(2 / 8) # This kind of initialization is called He Initialization
b2 = np.zeros(8)

W3 = np.random.randn(8, 1) * np.sqrt(2 / 8) # This is good for Relu activation function
b3 = np.zeros(1)

learning_rate = 0.001
mini_batch = 32

# The main loop:

for epoch in range(2000):

    # Shuffle training data
    indices = np.random.permutation(len(X_train_scaled))
    X_shuffled = X_train_scaled[indices]
    y_shuffled = y_train[indices]

    # Go through mini-batches
    for start in range(0, len(X_train_scaled), mini_batch):

        end = start + mini_batch

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # =====================
        # FORWARD PASS
        # =====================

        A1 = X_batch @ W1 + b1
        C1 = relu(A1)

        A2 = C1 @ W2 + b2
        C2 = relu(A2)

        predicted = C2 @ W3 + b3


        # =====================
        # LOSS
        # =====================

        loss = np.mean((predicted - y_batch) ** 2)


        # =====================
        # BACKPROPAGATION
        # =====================

        d_predicted = 2 * (predicted - y_batch) / y_batch.shape[0]

        dW3 = C2.T @ d_predicted
        db3 = np.sum(d_predicted, axis=0)

        dC2 = d_predicted @ W3.T
        dA2 = dC2 * (A2 > 0)

        dW2 = C1.T @ dA2
        db2 = np.sum(dA2, axis=0)

        dC1 = dA2 @ W2.T
        dA1 = dC1 * (A1 > 0)

        dW1 = X_batch.T @ dA1
        db1 = np.sum(dA1, axis=0)


        # =====================
        # UPDATE WEIGHTS AND BIASES
        # =====================

        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1


    # Print epoch loss occasionally
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


# =========================
# TESTING
# =========================

A1 = X_test_scaled @ W1 + b1
output1 = relu(A1)

A2 = output1 @ W2 + b2
output2 = relu(A2)

predicted_test = output2 @ W3 + b3

test_loss = np.mean((predicted_test - y_test) ** 2)

print("Test MSE:", test_loss)   # Mean Squared Error kati error xa vanxa
print("Test RMSE:", np.sqrt(test_loss))  # Root mean squared error kati unit error xa ta in terms of actual data (after standardaziation)

for i in range(10):
    print(
        "Actual:", y_test[i][0],
        "Predicted:", predicted_test[i][0]
    )


# =========================
# GIVE THE MODEL A NEW HOUSE
# =========================

house = np.array([])

transaction_date = int(input("Enter the last transaction date of the house format --> 2012.5 --> mid of 2012"))
house_age = int(input("Please enter the House age in years"))
distance_mrt = int(input("Please enter the distance to the market/mart from the house in meters"))
stores = int(input("Please enter the number of stores nearby"))
latitude = int(input("Please enter latitude of the house"))
longitude = int(input("Please enter the longitude of the house"))

house = np.append(house, house_age)
house = np.append(house, distance_mrt)
house = np.append(house, stores)
house = np.append(house, latitude)
house = np.append(house, longitude)


# Use the SAME mean and SD calculated from X_train
house_scaled = (house - train_mean) / train_sd

# =========================
# FORWARD PASS
# =========================

A1 = house_scaled @ W1 + b1
output1 = relu(A1)

A2 = output1 @ W2 + b2
output2 = relu(A2)

predicted_price = output2 @ W3 + b3

print("Predicted house price:", predicted_price[0, 0])
















