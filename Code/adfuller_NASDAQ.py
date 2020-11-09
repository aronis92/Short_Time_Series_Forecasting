from functions.utils import adfuller_test
from math import log
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


np.random.seed(0)
n = 300
X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = n)
X = X.to_numpy()
X = X.T


X_norm = np.zeros(X.shape)

for r in range(X.shape[0]):
    m = np.mean(X[r, ...])
    std = np.std(X[r, ...])
    X_norm[r, ...] = ( X[r, ...] - m )/std


plt.figure(figsize = (12,5))
plt.plot(X_norm.T)
plt.show()


df = pd.DataFrame(X_norm[..., 160:200].T)

plt.figure(figsize = (12,5))
plt.plot(X_norm[..., 160:200].T)
plt.show()





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
print("NASDAQ original: ", counter, "/", X.shape[0], "Series are Stationary")
#for i in indices:
#    print("Non-stationarity at index: ", i)







# Differencing

X_d = 0
for i in range(d):
    tmp = X[..., (d-i):(n_total*2-i)] - X[..., (d-i-1):(n_total*2-i-1)]
    X_d += (-1)**i*tmp

X_d = X_d[..., -n_total:]

    
    
    
    
    
    