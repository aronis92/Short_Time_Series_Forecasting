import pandas as pd
import numpy as np
from fbprophet import Prophet
from statsmodels.tsa.arima_process import arma_generate_sample
import time

def compute_rmse(y_pred, y_true):
    rmse = np.sqrt( np.linalg.norm(y_pred - y_true)**2 / np.size(y_true) )
    return rmse

def compute_nrmse(y_pred, y_true):
    # t1 = np.linalg.norm(y_pred - y_true)**2 / np.size(y_true)
    t1 = compute_rmse(y_pred, y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    nrmse = t1 / t2
    return nrmse

def book_data(sample_size):
    np.random.seed(69)
    A1 = np.array([[.3, -.2, .04], [-.11, .26, -.05], [.08, -.39, .39]])
    A2 = np.array([[.28, -.08, .07], [-.04, .36, -.1], [-.33, .05, .38]])
    total = sample_size + 2000
    X_total = np.zeros((3, total))
    X_total[..., 0:2] = np.random.rand(3,2)
    for i in range(2, total):
        X_total[..., i] = np.dot(-A1, X_total[..., i-1]) + np.dot(-A2, X_total[..., i-2]) + np.random.rand(3,)
    return X_total[..., (total-sample_size):], A1, A2

n_train = 40
n_val = 5
n_test = 5
n_total = n_train + n_val + n_test

#data = pd.read_csv('/content/nasdaq100_padding.csv',  nrows = n_total)
#data = data.to_numpy()
#data = data.T
X, _, _ = book_data(sample_size=n_total)

X_train = np.append(X[..., :n_train], X[..., n_train:(n_val+n_train)], axis=-1)
X_train = X_train[..., n_val:]
X_test = X[..., -n_test:]

#import matplotlib.pyplot as plt
#plt.figure(figsize=(12,5))
#plt.ylim(-1, 2)
#plt.plot(X_train.T)
#plt.show()

X_train = pd.DataFrame(X_train.T)
#print(X_train.header())
ds = pd.date_range('2015-02-24', periods = n_train, freq='D')
ds = pd.DataFrame(ds.date)
ds = ds.rename(columns={0:'ds'})
new_data = pd.concat([ds, X_train], axis = 1)
print(new_data.tail())
print(new_data.shape)




