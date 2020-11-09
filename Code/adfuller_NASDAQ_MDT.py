from functions.utils import adfuller_test
from functions.MDT_functions import MDT
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd
import numpy as np
import time


np.random.seed(0)
r = 3
d = 1
n = 200
X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = n)
X = X.to_numpy()
X = X.T
X = X[..., (160-d):]

X_norm = np.zeros(X.shape)

for i in range(X.shape[0]):
    m = np.mean(X[i, ...])
    std = np.std(X[i, ...])
    X_norm[i, ...] = ( X[i, ...] - m )/std

X_hat, _ = MDT(X_norm, r)
X_hat_vec = tl.unfold(X_hat, -1).T
df_hat = pd.DataFrame(X_hat_vec).T

plt.figure(figsize = (12,5))
plt.plot(X_hat_vec[0,...].T)

counter = 0
indices = []
index = 0
for name, column in df_hat.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    #print('\n')
print("NASDAQ - MDT: ", counter, "/", X_hat_vec.shape[0], "Series are Stationary")
#for i in indices:
#    print("Non-stationarity at index: ", i)




    
    
    
    
    