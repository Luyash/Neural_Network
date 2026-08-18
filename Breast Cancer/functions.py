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

