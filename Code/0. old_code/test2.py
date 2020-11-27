from functions.utils import book_data
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from numpy import linalg as la
from numpy import log



def book_data(sample_size):
    np.random.seed(69)
    #c = np.array([56.1, 49.9, 59.6])
    A1 = np.array([[.3, -.2, .04],
                    [-.11, .26, -.05],
                    [.08, -.39, .39]])
    #print(la.norm(A_1, 'fro'))
    A2 = np.array([[.28, -.08, .07],
                    [-.04, .36, -.1],
                    [-.33, .05, .38]])
    #print(la.norm(A_2, 'fro'))
    total = 2000
    
    X_total = np.zeros((3, total))
    X_total[..., 0:2] = np.random.rand(3,2)
    for i in range(2, total):
        X_total[..., i] = np.dot(A1, X_total[..., i-1]) + np.dot(A2, X_total[..., i-2]) + np.random.rand(3,)
        
    return X_total[..., (total-sample_size):], A1, A2#, c


def get_matrix_coeff_data(sample_size, n_rows, n_columns):
    np.random.seed(42)
    seed(42)
    total = 2000

    X_total = np.zeros((n_rows*n_columns, total))
    X_total[..., 0:2] = log(np.random.rand(n_rows*n_columns, 2))
    max_v = 2.5
    min_v = 1.5
    A1 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    A1 = A1/((min_v + random()*(max_v - min_v))*la.norm(A1, 'fro'))
    A2 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    A2 = A2/((min_v + random()*(max_v - min_v))*la.norm(A2, 'fro'))

    for i in range(2, total):
        X_total[..., i] = np.dot(A1, X_total[..., i-1]) + np.dot(A2, X_total[..., i-2]) + np.random.rand(n_rows*n_columns)
    
    
    X = X_total[..., (total-sample_size):]
    return X, A1, A2


def estimate_matrix_coefficients(data, p):
    k = data.shape[0]
    T = data.shape[1]
    Y = data[:, p:]
    Z = np.zeros((k*p+1, T-p))
    Z[0,:] = 1
    # print("k: ", k)
    # print("Z :", Z.shape)
    # print("X :", X.shape)
    # print("Y :", Y.shape)    
    
    for i in range(1, p+1):
        Z[(k*(i-1)+1):(k*i+1), :] = data[:, (p-i):(-i)]
        
    B = np.linalg.multi_dot([Y, Z.T, np.linalg.inv(np.dot(Z, Z.T))])

    c = B[:, 0]
    A = [c]
    for i in range(p):
        A.append(B[:, (k*i+1):(k*i+1+k)])
    
    return A
        

# X, A1, A2 = book_data(sample_size=40)
X, A1, A2 = get_matrix_coeff_data(sample_size=10, n_rows=3, n_columns=3)

#plt.figure(figsize = (16,5))
#X_vectorized = X_vectorized[5, :]
#plt.plot(X.T)

A = estimate_matrix_coefficients(X, p=2)











