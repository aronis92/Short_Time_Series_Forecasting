from functions.utils import adfuller_test
import matplotlib.pyplot as plt
from random import random, seed
from numpy import linalg as la
from math import log
import numpy as np
import pandas as pd

def semi_definite(matrixSize):
    from scipy import random, linalg
    A = random.rand(matrixSize,matrixSize)
    B = np.dot(A,A.transpose())
    return B

def get_matrix_coeff_data(sample_size, n_rows, n_columns):
    np.random.seed(42)
    seed(42)
    total = sample_size + 20
    
    mean = np.zeros(n_rows*n_columns)
    cov = semi_definite(n_rows*n_columns)
    X_total = np.zeros((n_rows*n_columns, total))
    
    X_total[..., 0:2] = np.random.multivariate_normal(mean, cov, 2).T
    max_v = 3
    min_v = 2.5
    A1 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    #A1 = A1/((min_v + random()*(max_v - min_v))*la.norm(A1, 'fro'))
    A1 = A1/(3*la.norm(A1, 'fro'))
    A2 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    #A2 = A2/((min_v + random()*(max_v - min_v))*la.norm(A2, 'fro'))
    A2 = A2/(3*la.norm(A2, 'fro'))
    for i in range(2, total):
        X_total[..., i] = np.dot(A1, X_total[..., i-1]) + np.dot(A2, X_total[..., i-2]) + np.random.rand(n_rows*n_columns)
    
    
    X = X_total[..., (total-sample_size):]
    return X, A1, A2

X, _, _ = get_matrix_coeff_data(sample_size=50, n_rows=3, n_columns=3)

plt.figure(figsize = (12,5))
#plt.ylim(-0.001, 0.001)
plt.plot(X.T)
plt.show()

print(np.mean(X, axis=1))
df = pd.DataFrame(X.T)
counter = 0
indices = []
index = 0
for name, column in df.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    #print('\n')
print(counter, "/", X.shape[0], "Series are Stationary")




