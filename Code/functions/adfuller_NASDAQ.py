from utils import adfuller_test, get_data, difference, inv_difference
import matplotlib.pyplot as plt
from MDT_functions import MDT
import tensorly as tl
import pandas as pd
import numpy as np
import time

# train = 50, val = 5, test = 5
# d = 1, 2, 3 does not provide stationarity for original time series
# d = 4 does not provide stationarity for all MDT orders r = [2, 10]
# d = 5, 6, 7 provides stationarity for original Series and MDT orders r = [2, 10]


def test_stationarity(X):
    
    for d in range(4, 8):
        X, _, _ = get_data(dataset = "nasdaq",
                           Ns = [60 + d, 5, 5])
        print("X shape:", X.shape)
  
        '''Check differenced data for Stationarity'''
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
        print(counter, "/", X.shape[0], "of the original Series are Stationary After ",d,"-order Differencing")

        '''Check various MDT orders for Stationarity'''
        # d = 6
        # X, _, _ = get_data(dataset = "nasdaq",
        #                    Ns = [60 + d, 5, 5])
        
        X_d, inv = difference(X, d)
        for r in range(2, 11):
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
            print("r:", r, "d:", d, "~", counter, "/", X_hat_vec.shape[0], "Series are Stationary")



'''Load the data'''
X, _, _ = get_data(dataset = "nasdaq",
                   Ns = [60, 5, 5])

# plt.figure(figsize = (12,5))
# plt.plot(X[1:3,...].T)
# plt.show()

'''Check if Series are Stationary'''
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
print(counter, "/", X.shape[0], "of the original Series are Stationary")

test_stationarity(X)







    
    
    
    
    
    