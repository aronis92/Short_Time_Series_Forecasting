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
n = 40 + d
X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = n)
X = X.to_numpy()
X = X.T

X_hat, _ = MDT(X, r)
X_hat_vec = tl.unfold(X_hat, -1).T
df_hat = pd.DataFrame(X_hat_vec).T

# plt.figure(figsize = (12,5))
# plt.plot(X[0,...].T)

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




    
    
    
    
    