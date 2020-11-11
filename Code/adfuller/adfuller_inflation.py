from functions.utils import adfuller_test, get_data, difference, inv_difference
from functions.MDT_functions import MDT
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd
import numpy as np
import time

# train = 50, val = 5, test = 5
# d = 5, 6 provides stationarity for series and mdt for r = [2, 12]
# d = 4 provides stationarity for r = 2, 3
# d = 1, 2, 3 does not provide stationarity for original time series nor mdt

X, _, _ = get_data(dataset = "inflation",
                   Ns = [60, 5, 5])

counter = 0
indices = []
index = 0
df_train = pd.DataFrame(X.T)
for name, column in df_train.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    #print('\n')
print(counter, "/", X.shape[0], "Original Series are Stationary")


for d in range(1, 7):
    X_d, inv = difference(X, d)
    df_differenced = pd.DataFrame(X_d.T)
    counter = 0
    indices = []
    index = 0
    for name, column in df_differenced.iteritems():
        c = adfuller_test(column, name=column.name)
        if c==0:
            indices.append(index)
        index += 1
        counter += c
        #print('\n')
    print(counter, "/", X.shape[0], "Original Series are Stationary After ",d,"-order Differencing")


X_d, inv = difference(X, 4)
for r in range(2, 13):
    X_hat, _ = MDT(X_d, r)
    X_hat_vec = tl.unfold(X_hat, -1).T
    df_hat = pd.DataFrame(X_hat_vec.T)
    
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
    print("MDT of order", r, "~", counter, "/", X_hat_vec.shape[0], "Series are Stationary")












