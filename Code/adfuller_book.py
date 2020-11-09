from functions.utils import adfuller_test, book_data, get_matrix_coeff_data
from random import random, seed
import matplotlib.pyplot as plt
from numpy import linalg as la
from numpy import log
import pandas as pd
import numpy as np



np.random.seed(0)
d = 1
n_train = 40 + d

X, _, _ = book_data(sample_size=n_total)


# plt.figure(figsize = (12,5))
# #plt.ylim(-1, 1.7)
# plt.plot(X[2,...].T)


df_train = pd.DataFrame(X.T)



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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    