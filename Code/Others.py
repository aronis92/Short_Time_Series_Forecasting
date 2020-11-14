#################################################
##                                             ##
##  This file contains functions that execute  ##
##  other models and return their performance  ##
##                                             ##
#################################################

from functions.utils import compute_nrmse, compute_rmse
from functions.utils import get_data
from functions.AR_functions import fit_ar, estimate_matrix_coefficients
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
np.random.seed(0)



def VAR_results(data_train, data_val, p):
    """
    The function that calculates and returns the 
    results of vector AR with matrix coefficients
    
    Input:
        data: The data matrix
        p: AR model order
        
    Returns:
        A: The coefficient matrices as a list
        duration: The total time of training
        rmse: The RMSE for the predicted value
        nrmse: The NRMSE for the predicted value
    """
    start = time.clock()
    model = VAR(data_train.T)
    results = model.fit(p)
    end = time.clock()
    A = results.coefs
    
    predictions = results.forecast(data_train[..., -p:].T, data_val.shape[-1])

    duration = end - start
    rmse = compute_rmse(predictions.T, data_val)
    nrmse = compute_nrmse(predictions.T, data_val)
    return A, duration, rmse, nrmse


def AR_results(data_train, data_val, data_test, p):
    """
    The function that calculates and returns the 
    results of vector AR with scalar coefficients
    
    Input:
        data: The data matrix
        p: AR model order
    
    Returns:
        A: The coefficients as a list
        duration: The total time of training
        rmse: The RMSE for the predicted value
        nrmse: The NRMSE for the predicted value
    """
    data_train2 = np.append(data_train, data_val, axis=-1)
    data_train2 = data_train2[..., data_val.shape[-1]:]
    
    start = time.clock()
    A = fit_ar(data_train2, p)
    end = time.clock()
    
    predictions = np.zeros(data_test.shape)
    predictions = np.append(data_train2[..., -p:], predictions, axis=-1)
    
    for i in range(p, p+data_test.shape[-1]):
        for j in range(p):
            predictions[..., i] += A[j]*predictions[..., i-j-1]
    
    duration = end - start
    rmse = compute_rmse(predictions[..., p:], data_test)
    nrmse = compute_nrmse(predictions[..., p:], data_test)
    return A, duration, rmse, nrmse




'''Create/Load Dataset'''
X_train, X_val, X_test = get_data(dataset = "book", Ns = [50, 5, 5])


fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
# fig.suptitle('Sharing both axes')
axs[0].plot(X_train.T)
axs[1].plot(X_train[1, :].T, 'tab:orange')
axs[2].plot(X_train[0, :].T, 'tab:blue')
axs[3].plot(X_train[2, :].T, 'tab:green')
plt.show()




var_A, var_duration, var_rmse, var_nrmse = VAR_results(data_train = np.append(X_train, X_val, axis=-1), 
                                                       data_val = X_test, 
                                                       p = 2)

ar_A, ar_duration, ar_rmse, ar_nrmse,  = AR_results(data_train = X_train, 
                                                    data_val = X_val,
                                                    data_test = X_test,
                                                    p=2)

print("RMSE AR: ", ar_rmse)
print("NRMSE AR: ", ar_nrmse)
# print("Duration AR: ", ar_duration)
print("RMSE VAR: ", var_rmse)
print("NRMSE VAR: ", var_nrmse)
# print("Duration VAR: ", var_duration)







