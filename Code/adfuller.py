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


def book_data(sample_size):
    np.random.seed(69)
    A1 = np.array([[.3, -.2, .04],
                   [-.11, .26, -.05],
                   [.08, -.39, .39]])
    #print(la.norm(A_1, 'fro'))
    A2 = np.array([[.28, -.08, .07],
                   [-.04, .36, -.1],
                   [-.33, .05, .38]])
    #print(la.norm(A_2, 'fro'))
    total = sample_size + 1000
    
    X_total = np.zeros((3, total))
    X_total[..., 0:2] = np.random.rand(3,2)
    for i in range(2, total):
        X_total[..., i] = np.dot(-A1, X_total[..., i-1]) + np.dot(-A2, X_total[..., i-2]) + np.random.rand(3,)
        
    return X_total[..., (total-sample_size):], A1, A2


def get_matrix_coeff_data(sample_size, n_rows, n_columns):
    np.random.seed(42)
    seed(42)
    total = 2000

    X_total = np.zeros((n_rows*n_columns, total))
    X_total[..., 0:2] = log(np.random.rand(n_rows*n_columns, 2))
    max_v = 2.5
    min_v = 1.5
    A1 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    A1 = A1/((min_v + random()*(max_v - min_v))*la.norm(A1, 'fro'))
    A2 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    A2 = A2/((min_v + random()*(max_v - min_v))*la.norm(A2, 'fro'))

    for i in range(2, total):
        X_total[..., i] = np.dot(A1, X_total[..., i-1]) + np.dot(A2, X_total[..., i-2]) + np.random.rand(n_rows*n_columns)
    
    
    X = X_total[..., (total-sample_size):]
    return X, A1, A2


np.random.seed(0)
n_train = 40
n_val = 5
n_test = 5
n_total = n_train + n_val + n_test

X, _, _ = book_data(sample_size=n_total)
#X, _, _ = get_matrix_coeff_data(sample_size = n_total, n_rows=3, n_columns=2)


plt.figure(figsize = (12,5))
plt.ylim(-1, 1.7)
plt.plot(X.T)


X_train = X[..., :n_train]
X_val = X[..., n_train:(n_val+n_train)]
X_test = X[..., -n_test:]
#tmp = np.append(X_train, X_val, axis=-1)[..., n_val:].T

df_train = pd.DataFrame(X_train.T)
#df_train = pd.DataFrame(tmp)
df_val = pd.DataFrame(X_val)
df_test = pd.DataFrame(X_test)

# ADF Test on each column
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    