from functions.utils import adfuller_test, book_data
from functions.MDT_functions import MDT
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd
import numpy as np


np.random.seed(0)
r = 5 # 1-10 OK
d = 1
n_total = 40

X, _, _ = book_data(sample_size=n_total)
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
# for i in indices:
#     print("Non-stationarity at index: ", i)


X_hat, _ = MDT(X, r)
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
print(counter, "/", X_hat_vec.shape[0], "Series are Stationary")
# for i in indices:
#     print("Non-stationarity at index: ", i)
    
    
# Differencing
# d = 1
# X_d = 0
# for i in range(d):
#     tmp = X_hat_vec[..., (d-i):(n_total*2-i)] - X_hat_vec[..., (d-i-1):(n_total*2-i-1)]
#     X_d += (-1)**i*tmp

# X_d = X_d[..., -n_total:]
    
    
    
    
    
    
    
    
    
    
    
    
    