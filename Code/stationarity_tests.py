from functions.utils import adfuller_test, get_data, difference, inv_difference
from functions.MDT_functions import MDT
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd
import numpy as np
import time


def stationarity_tests(dataset_name, ds, rs, N):
    X, _, _ = get_data(dataset = dataset_name, Ns = [N, 1, 1])
    
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
    print("Original Time Series")
    print(counter, "/", X.shape[0], "are Stationary")
    

    print("\nHankelized Original Time Series")
    for r in rs:
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
        
        print("r:", r, "~", counter, "/", X_hat_vec.shape[0], "Series are Stationary")
    
    
    for d in ds:
        X, _, _ = get_data(dataset = dataset_name,  Ns = [N + d, 1, 1])
        #print("X shape:", X.shape)
  
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
        print("\nOriginal Time Series after",d,"-order Differencing")
        print(counter, "/", X.shape[0], "are Stationary")

        '''Check various MDT orders for Stationarity'''       
        X_d, inv = difference(X, d)
        print("\nHankelized Time Series on", d, "-order differenced data")
        for r in rs:
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
            print("r:", r, "~", counter, "/", X_hat_vec.shape[0], "Series are Stationary")



# ds = [i for i in range(1, 8)]
# rs = [i for i in range(2, 11)]
# print("Dataset: Book \n")
# stationarity_tests("book", ds, rs, N=70)
# print("____________________________\n")

# ds = [i for i in range(1, 8)]
# rs = [i for i in range(2, 11)]
# print("Dataset: NASDAQ \n")
# stationarity_tests("nasdaq", ds, rs, N=70)
# print("____________________________\n")

# ds = [i for i in range(1, 8)]
# rs = [i for i in range(2, 11)]
# print("Dataset: Inflation \n")
# stationarity_tests("inflation", ds, rs, N=70)
# print("____________________________\n")

ds = [i for i in range(1, 8)]
rs = [i for i in range(2, 11)]
print("Dataset: Yahoo \n")
stationarity_tests("yahoo", ds, rs, N=52)
print("____________________________\n")















