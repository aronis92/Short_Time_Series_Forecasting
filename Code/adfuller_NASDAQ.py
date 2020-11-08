from functions.utils import get_matrix_coeff_data
from statsmodels.tsa.stattools import adfuller
from random import random, seed
import matplotlib.pyplot as plt
from numpy import linalg as la
from numpy import log
import pandas as pd
import numpy as np
import time

def adfuller_test(series, signif=0.05, name='', verbose=False):
    counter = 0
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
        counter = 1
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.") 
        counter = 0
    return counter



np.random.seed(0)
n_train = 40
n_val = 5
n_test = 5
n_total = n_train + n_val + n_test

X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = n_total*2)
X = X.to_numpy()
X = X.T

# Differencing
d = 2
X_d = 0
for i in range(d):
    tmp = X[..., (d-i):(n_total*2-i)] - X[..., (d-i-1):(n_total*2-i-1)]
    X_d += (-1)**i*tmp

X_d = X_d[..., -n_total:]

# plt.figure(figsize = (12,5))
# #plt.ylim(-1, 2)
# plt.plot(X_d[0:3].T)


X_train = X_d[..., :n_train]
X_val = X[..., n_train:(n_val+n_train)]
X_test = X[..., -n_test:]

df_train = pd.DataFrame(X_train.T)
df_val = pd.DataFrame(X_val)
df_test = pd.DataFrame(X_test)

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



# Remove the non-stationary indices
X_train = np.delete(X_train, indices, 0)
X_val = np.delete(X_val, indices, 0)
X_test = np.delete(X_test, indices, 0)
    
df_train = pd.DataFrame(X_train.T)
df_val = pd.DataFrame(X_val)
df_test = pd.DataFrame(X_test)
 
counter = 0
for name, column in df_train.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    print('\n')
print(counter, "Series are Stationary")
    
    
    
    
    
    