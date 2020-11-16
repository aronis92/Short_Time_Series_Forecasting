from functions.utils import get_data, adfuller_test, difference
from functions.MDT_functions import MDT
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd

def stationarity_tests(dataset_name, ds, rs, Ns):
    X, _, _ = get_data(dataset_name, Ns)
    
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
        X, _, _ = get_data(dataset_name,  Ns)
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

#                             Index  Var x Time
datasets = ['macro', #__________0     12 x 203
            'elnino', #_________1     12 x 61
            'copper', #_________2      5 x 25    # DROP
            'fertility', #______3    192 x 52    # Maybe drop
            'stackloss', #______4      4 x 21
            'nightvisitors', #__5      8 x 56
            'mortality', #______6      2 x 72
            'ozone', #__________7      8 x 203
            'inflation', #______8      8 x 123  
            'nasdaq', #_________9     82 x 40560 # Pending
            'traffic', #________10   228 x 40    # DROP
            'yahoo', #__________11     5 x 2469  
            'book'] #___________12     3 x sum(Ns)

Ns = [1000, 1, 1]
data_name = datasets[10]

X, _, _ = get_data(data_name, Ns)


plt.figure(figsize = (12,5))
plt.plot(X.T)
plt.show()



ds = [i for i in range(8, 10)]
rs = [i for i in range(2, 11)]
print("Dataset: " + data_name + " \n")
stationarity_tests(data_name, ds, rs, Ns)






