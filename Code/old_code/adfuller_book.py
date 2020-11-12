from functions.utils import adfuller_test, get_data
from functions.MDT_functions import MDT
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd
import numpy as np
np.random.seed(0)

# No differencing
# train = 50, val = 5, test = 5
# Stationarity for r = [2, 12]

def test_stationarity(X):
    counter = 0
    indices = []
    index = 0
    df = pd.DataFrame(X.T)
    for name, column in df.iteritems():
        c = adfuller_test(column, name=column.name)
        if c==0:
            indices.append(index)
        index += 1
        counter += c
        #print('\n')
    print(counter, "/", X.shape[0], "Original Series are Stationary")
    # for i in indices:
    #     print("Non-stationarity at index: ", i)
    
    for r in range(2, 13):
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
        print("MDT of order", r, "~", counter, "/", X_hat_vec.shape[0], "Series are Stationary")
    # for i in indices:
    #     print("Non-stationarity at index: ", i)
    
X, _, _ = get_data(dataset = "book",
                   Ns = [60, 5, 5])

test_stationarity(X)
    
    
    
    
    
    
    
    
    
    
    