import numpy as np
from functions.utils import difference, inv_difference


d = 3
X = np.array([[1, 3, 4, 7, 9],
              [2, 5, 3, 1, 4]])

X, inv = difference(X, d)

X_r = inv_difference(X, inv, d)



# inv = []
# for _ in range(d):
#     inv.append(X[:, 0])
#     X = np.diff(X)

# X_r = X
# for _ in range(d):
#     tmp = inv.pop().reshape((X.shape[0],1))
#     X_r = np.cumsum(np.hstack([tmp, X_r]), axis=-1)
    
