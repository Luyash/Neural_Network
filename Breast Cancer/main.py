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


