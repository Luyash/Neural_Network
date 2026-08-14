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




