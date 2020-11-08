#################################################
##                                             ##
##  This file contains functions that execute  ##
##  other models and return their performance  ##
##                                             ##
#################################################

from functions.utils import compute_nrmse, compute_rmse
from functions.utils import get_matrix_coeff_data, create_synthetic_data2, book_data
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


# The function that calculates and returns the 
# results of vector AR with matrix coefficients
# Input:
#   data: The data matrix
#   p: AR model order
# Returns:
#   A: The coefficient matrices as a list
#   duration: The total time of training
#   rmse: The RMSE for the predicted value
#   nrmse: The NRMSE for the predicted value
def VAR_results(data, p):
    start = time.clock()
    model = VAR(data[..., :-1].T)
    results = model.fit(p)

    A = results.coefs
    end = time.clock()
    prediction = results.forecast(data[..., -p:].T, 1)

    duration = end - start
    rmse = compute_rmse(prediction, data[..., -1])
    nrmse = compute_nrmse(prediction, data[..., -1])
    return A, duration, rmse, nrmse


# The function that calculates and returns the 
# results of vector AR with scalar coefficients
# Input:
#   data: The data matrix
#   p: AR model order
# Returns:
#   A: The coefficients as a list
#   duration: The total time of training
#   rmse: The RMSE for the predicted value
#   nrmse: The NRMSE for the predicted value
def ARIMA_results(data_train, data_val, data_test, p):
    
    data_train2 = np.append(data_train, data_val, axis=-1)
    data_train2 = data_train2[..., data_val.shape[-1]:]
    
    prediction = np.zeros([1, data_train.shape[0]])
    start = time.clock()
    alpha = 0
    for i in range(data_train2.shape[0]):
        model = ARIMA(data_train2[i, :-1].T, order=(p,0, 0))
        results = model.fit()
        prediction[0, i] = results.forecast()
        alpha += results.polynomial_ar
    end = time.clock()

    duration = end - start
    rmse = compute_rmse(prediction, data_train2[..., -1])
    nrmse = compute_nrmse(prediction, data_train2[..., -1])
    #print(data_train2.shape[0])
    print(-alpha[1:])
    A = -alpha[1:]/data_train2.shape[0]
    return A, duration, rmse, nrmse


# Create/Load Dataset
np.random.seed(0)
n_train = 40
n_val = 5
n_test = 5
n_total = n_train + n_val + n_test

# X = create_synthetic_data2(p = 2, dim = 10, n_samples=6)
# X = create_synthetic_data(p = 2, dim = 100, n_samples=40)
# X, A1, A2 = get_matrix_coeff_data(sample_size=500, n_rows=3, n_columns=3)
X, A1, A2 = book_data(sample_size=n_total)
# X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = 6)
# X = X.to_numpy()
# X = X.T

X_train = X[..., :n_train]
X_val = X[..., n_train:(n_val+n_train)]
X_test = X[..., -n_test:]

# var_A, var_duration, var_rmse, var_nrmse = VAR_results(data_train=X,
#                                                        data_val = X_test,
#                                                        p=2)

ar_A, ar_duration, ar_rmse, ar_nrmse,  = ARIMA_results(data_train = X_train, 
                                                       data_val = X_val,
                                                       data_test = X_test,
                                                       p=2)

# print("RMSE AR: ", ar_rmse)
print("NRMSE AR: ", ar_nrmse)
# print("Duration AR: ", ar_duration)
# print("RMSE VAR: ", var_rmse)
#print("NRMSE VAR: ", var_nrmse)
# print("Duration VAR: ", var_duration)





# from test2 import estimate_matrix_coefficients
# myA = estimate_matrix_coefficients(X, 2)


# from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
# from functions.ARIMA_functions import fit_ar, autocorrelation

# import scipy as sp
# import copy

# del arima_duration, arima_rmse, arima_nrmse
# model = VARMAX(X[..., :-1], order = (2, 0), enforce_stationarity=False) # 
# model_fit = model.fit(maxiter = 100, disp = False)