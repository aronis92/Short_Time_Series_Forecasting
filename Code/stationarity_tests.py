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
datasets = ['macro', #__________0     12 x 203     #
            'elnino', #_________1     12 x 61      #
            'stackloss', #______2      4 x 21      #
            'nightvisitors', #__3      8 x 56      #
            'ozone', #__________4      8 x 203     #
            'nasdaq', #_________5     82 x 40560   # Pending
            'yahoo', #__________6      5 x 2469    #
            'book1'] #__________7      3 x sum(Ns) #

Ns = [20, 1, 1]
data_name = datasets[7]

X_train, _, _ = get_data(data_name, Ns)


# plt.figure(figsize = (12,5))
# plt.plot(X.T)
# plt.show()
fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
# fig.suptitle('Sharing both axes')
axs[0].plot(X_train.T)
axs[1].plot(X_train[0, :].T, 'tab:blue')
axs[2].plot(X_train[1, :].T, 'tab:orange')
axs[3].plot(X_train[2, :].T, 'tab:green')
plt.show()


ds = [i for i in range(1, 9)]
rs = [i for i in range(2, 11)]
print("Dataset: " + data_name + " \n")
stationarity_tests(data_name, ds, rs, Ns)






