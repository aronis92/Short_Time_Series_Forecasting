#################################################
##                                             ##
##  This file contains functions that execute  ##
##  other models and return their performance  ##
##                                             ##
#################################################

from functions.utils import compute_nrmse, compute_rmse
from functions.utils import get_data, difference, inv_difference
from functions.AR_functions import fit_ar, estimate_matrix_coefficients
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import copy
np.random.seed(0)



def VAR_results(data_train, data_val, data_test, p, d):
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
    '''
    model = VAR(data_train.T)
    results = model.fit(p)
    A = results.coefs
    predictions_val = results.forecast(data_train[..., -p:].T, data_val.shape[-1])
    rmse = compute_rmse(predictions_val.T, data_val)
    nrmse = compute_nrmse(predictions_val.T, data_val)
    results_val = [rmse, nrmse]
    print(results_val)
    '''
    if d>0:
        data_train_original = copy.deepcopy(data_train)
        data_train, inv = difference(data_train, d)
    
    start = time.clock()
    A = estimate_matrix_coefficients(data_train, p)
    end = time.clock()
    duration = end - start
    
    # Create the validation prediction array
    # The last p elements of data_train concatened 
    # with zeroes for the values to be predicted
    predictions_val = np.append(data_train[..., -p:], np.zeros(data_val.shape), axis=-1)
    
    # Forecast the next values
    for i in range(p, p+data_val.shape[-1]):
        predictions_val[..., i] += A[0]
        for j in range(p):
            predictions_val[..., i] += np.dot(A[j+1], predictions_val[..., i-j-1])
    
    predictions_val = predictions_val[..., -data_val.shape[-1]:]
    
    tmp = np.append(data_train, predictions_val, axis=-1)    
    
    if d>0:
        predictions_val = inv_difference(tmp, inv, d)
    
    rmse = compute_rmse(predictions_val[..., -data_val.shape[-1]:], data_val)
    nrmse = compute_nrmse(predictions_val[..., -data_val.shape[-1]:], data_val)
    results_val = [rmse, nrmse]


    
    
    data_test_start = np.append(data_train, data_val, axis=-1)
    predictions_test = np.zeros(data_test.shape)
    predictions_test = np.append(data_test_start[..., -p:], predictions_test, axis=-1)
    
    results_test = []
    
    return results_val, results_test, duration


def AR_results(data_train, data_val, data_test, p, d):
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
    # Fit with the training data
    start = time.clock()
    A = fit_ar(data_train, p)
    end = time.clock()
    duration = end - start
    
    # Create the validation prediction array
    # The last p elements of data_train concatened 
    # with zeroes for the values to be predicted
    predictions_val = np.append(data_train[..., -p:], np.zeros(data_val.shape), axis=-1)
    
    # Forecast the next values
    for i in range(p, p+data_val.shape[-1]):
        for j in range(p):
            predictions_val[..., i] += A[j]*predictions_val[..., i-j-1]
    
    # Compute the RMSE and NRMSE for the validation set
    rmse_val = compute_rmse(predictions_val[..., p:], data_val)
    nrmse_val = compute_nrmse(predictions_val[..., p:], data_val)
    results_val = [rmse_val, nrmse_val]
    
    # Create the test prediction array
    # The last p elements of data_test_start 
    # with zeroes for the values to be predicted
    data_test_start = np.append(data_train, data_val, axis=-1)
    predictions_test = np.zeros(data_test.shape)
    predictions_test = np.append(data_test_start[..., -p:], predictions_test, axis=-1)
    
    # Forecast the next values
    for i in range(p, p+data_test.shape[-1]):
        for j in range(p):
            predictions_test[..., i] += A[j]*predictions_test[..., i-j-1]
    
    # Compute the RMSE and NRMSE for the test set
    rmse_test = compute_rmse(predictions_test[..., p:], data_test)
    nrmse_test = compute_nrmse(predictions_test[..., p:], data_test)
    results_test = [rmse_test, nrmse_test]
    
    return results_val, results_test, duration




'''Create/Load Dataset'''
X_train, X_val, X_test = get_data(dataset = "book", Ns = [50, 1, 1])


# X_train, _ = difference(X_train, 2)

# fig = plt.figure(figsize = (12,9))
# gs = fig.add_gridspec(4, hspace=0)
# axs = gs.subplots(sharex=True, sharey=False)
# # fig.suptitle('Sharing both axes')
# axs[0].plot(X_train.T)
# axs[1].plot(X_train[1, :].T, 'tab:orange')
# axs[2].plot(X_train[0, :].T, 'tab:blue')
# axs[3].plot(X_train[2, :].T, 'tab:green')
# plt.show()


ar_results_val, ar_results_test, ar_duration = AR_results(data_train = X_train, 
                                                          data_val = X_val,
                                                          data_test = X_test,
                                                          p = 2,
                                                          d = 1)

var_results_val, var_results_test, var_duration = VAR_results(data_train = X_train, 
                                                              data_val = X_val, 
                                                              data_test = X_test,
                                                              p = 2,
                                                              d = 3)

print("Autoregression with Scalar Coefficients")
print("Validation RMSE:  ", ar_results_val[0])
print("Validation NRMSE: ", ar_results_val[1])
print("Test RMSE:  ", ar_results_test[0])
print("Test NRMSE: ", ar_results_test[1])
# print("Duration AR: ", ar_duration)

print("\nAutoregression with Matrix Coefficients")
print("Validation RMSE:  ", var_results_val[0])
print("Validation NRMSE: ", var_results_val[1])
# print("Test RMSE:  ", var_results_test[0])
# print("Test NRMSE: ", var_results_test[1])
print("Duration VAR: ", var_duration)




