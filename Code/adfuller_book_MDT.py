from functions.MDT_functions import MDT
from random import random, seed
import matplotlib.pyplot as plt
from numpy import linalg as la
import tensorly as tl
from numpy import log
import pandas as pd
import numpy as np
from functions.utils import adfuller_test, book_data, get_matrix_coeff_data






np.random.seed(0)
n_train = 40
n_train_load = 50
n_val = 5
n_test = 5
n_total = n_train + n_val + n_test

#X, _, _ = book_data(sample_size=n_total)
X, _, _ = get_matrix_coeff_data(sample_size=2*n_total, n_rows=3, n_columns=2)


X_train = X[..., :n_train]
X_val = X[..., n_train:(n_val+n_train)]
X_test = X[..., -n_test:]

# plt.figure(figsize = (12,5))
# #plt.ylim(-1, 1.7)
# plt.plot(X_train[2,...].T)

X_hat, _ = MDT(X_train, 3)
X_hat_vec = tl.unfold(X_hat, -1).T


df_train = pd.DataFrame(X_train.T)
df_train_hat = pd.DataFrame(X_hat_vec.T)
# df_val = pd.DataFrame(X_val)
# df_test = pd.DataFrame(X_test)


# ADF Test on each column
counter = 0
indices = []
index = 0
for name, column in df_train_hat.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    print('\n')
print(counter, "Series are Stationary")
    
    
for i in indices:
    print("Non-stationarity at index: ", i)
    
    
# Differencing
d = 1
X_d = 0
for i in range(d):
    tmp = X_hat_vec[..., (d-i):(n_total*2-i)] - X_hat_vec[..., (d-i-1):(n_total*2-i-1)]
    X_d += (-1)**i*tmp

X_d = X_d[..., -n_total:]
    
    
    
    
    
    
    
    
    
    
    
    
    