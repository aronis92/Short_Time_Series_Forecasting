from functions.utils import adfuller_test
from random import random, seed
import matplotlib.pyplot as plt
from numpy import linalg as la
from numpy import log
import pandas as pd
import numpy as np
import time




np.random.seed(0)
n_train = 40
n_val = 5
n_test = 5
n_total = n_train + n_val + n_test

X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = n_total*2)
X = X.to_numpy()
X = X.T

# Differencing
d = 2
X_d = 0
for i in range(d):
    tmp = X[..., (d-i):(n_total*2-i)] - X[..., (d-i-1):(n_total*2-i-1)]
    X_d += (-1)**i*tmp

X_d = X_d[..., -n_total:]

# plt.figure(figsize = (12,5))
# #plt.ylim(-1, 2)
# plt.plot(X_d[0:3].T)


X_train = X_d[..., :n_train]
X_val = X[..., n_train:(n_val+n_train)]
X_test = X[..., -n_test:]

df_train = pd.DataFrame(X_train.T)
df_val = pd.DataFrame(X_val)
df_test = pd.DataFrame(X_test)

# ADF Test on each column
counter = 0
indices = []
index = 0
for name, column in df_train.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    print('\n')
print(counter, "Series are Stationary")
    
    
for i in indices:
    print("Non-stationarity at index: ", i)



# Remove the non-stationary indices
X_train = np.delete(X_train, indices, 0)
X_val = np.delete(X_val, indices, 0)
X_test = np.delete(X_test, indices, 0)
    
df_train = pd.DataFrame(X_train.T)
df_val = pd.DataFrame(X_val)
df_test = pd.DataFrame(X_test)
 
counter = 0
for name, column in df_train.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    print('\n')
print(counter, "Series are Stationary")
    
    
    
    
    
    