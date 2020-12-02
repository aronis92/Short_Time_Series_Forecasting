from functions.utils import difference, inv_difference
from functions.BHT_AR_functions import MDT
import numpy as np


X = np.array([[2, 5, 3, 7, 1, 2],
              [1, 3, 2, 6, 8, 5],
              [2, 5, 9, 0, 1, 3]])

X, _ = difference(X, 1)

X_hat, _ = MDT(X, 2)


X2 = np.array([[2, 5, 3, 7, 1, 2],
               [1, 3, 2, 6, 8, 5],
              [2, 5, 9, 0, 1, 3]])

X_hat2, _ = MDT(X2, 2)

X_hat2, _ = difference(X_hat2, 1)