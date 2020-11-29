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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
    
    if d>0:
        predictions_val = inv_difference(np.append(data_train, predictions_val, axis=-1) , inv, d)
    
    rmse = compute_rmse(predictions_val[..., -data_val.shape[-1]:], data_val)
    nrmse = compute_nrmse(predictions_val[..., -data_val.shape[-1]:], data_val)
    results_val = [rmse, nrmse]

    # Testing Phase
    
    
    if d>0:
        data_test_start = np.append(data_train_original, data_val, axis=-1)
        data_test_start, inv = difference(data_test_start, d)
    else:
        data_test_start = np.append(data_train, data_val, axis=-1)
      
    predictions_test = np.append(data_test_start[..., -p:], np.zeros(data_test.shape), axis=-1)
    
    # # Forecast the next values
    for i in range(p, p+data_test.shape[-1]):
        predictions_test[..., i] += A[0]
        for j in range(p):
            predictions_test[..., i] += np.dot(A[j+1], predictions_test[..., i-j-1])
    
    predictions_test = predictions_test[..., -data_test.shape[-1]:]
    
    if d>0:
        predictions_test = inv_difference(np.append(data_test_start, predictions_test, axis=-1) , inv, d)
    
    rmse = compute_rmse(predictions_test[..., -data_test.shape[-1]:], data_test)
    nrmse = compute_nrmse(predictions_test[..., -data_test.shape[-1]:], data_test)
    results_test = [rmse, nrmse]
    # results_test = []
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
    if d>0:
        data_train_original = copy.deepcopy(data_train)
        data_train, inv = difference(data_train, d)
        
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
    
    predictions_val = predictions_val[..., -data_val.shape[-1]:]
    
    if d>0:
        predictions_val = inv_difference(np.append(data_train, predictions_val, axis=-1) , inv, d)
        
    # Compute the RMSE and NRMSE for the validation set
    rmse_val = compute_rmse(predictions_val[..., -data_val.shape[-1]:], data_val)
    nrmse_val = compute_nrmse(predictions_val[..., -data_val.shape[-1]:], data_val)
    results_val = [rmse_val, nrmse_val]
    
    # Create the test prediction array
    # The last p elements of data_test_start 
    # with zeroes for the values to be predicted
    if d>0:
        data_test_start = np.append(data_train_original, data_val, axis=-1)
        data_test_start, inv = difference(data_test_start, d)
    else:
        data_test_start = np.append(data_train, data_val, axis=-1)
      
    predictions_test = np.append(data_test_start[..., -p:], np.zeros(data_test.shape), axis=-1)
    
    
    # data_test_start = np.append(data_train, data_val, axis=-1)
    # predictions_test = np.zeros(data_test.shape)
    # predictions_test = np.append(data_test_start[..., -p:], predictions_test, axis=-1)
    
    # Forecast the next values
    for i in range(p, p+data_test.shape[-1]):
        for j in range(p):
            predictions_test[..., i] += A[j]*predictions_test[..., i-j-1]
    
    # Compute the RMSE and NRMSE for the test set
    rmse_test = compute_rmse(predictions_test[..., p:], data_test)
    nrmse_test = compute_nrmse(predictions_test[..., p:], data_test)
    results_test = [rmse_test, nrmse_test]
    
    return results_val, results_test, duration



#                             Index  Var x Time
datasets = ['book1', #__________0     3 x sum(Ns) # DONE
            'stackloss', #______1     4 x 21      # DONE
            'macro', #__________2     12 x 203    # DONE
            'elnino', #_________3     12 x 61     # DONE
            'ozone', #__________4      8 x 203    # DONE
            'nightvisitors', #__5      8 x 56     # 
            'nasdaq', #_________6     82 x 40560  # 
            'yahoo'] #__________7     5 x 2469    #    

'''Create/Load Dataset'''
X_train, X_val, X_test = get_data(dataset = datasets[5], Ns = [54, 1, 1])
# data_train = X_train
# data_val = X_val
# data_test = X_test
# d = 1
# p = 2 

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

print("Autoregression with Scalar Coefficients")
min_v = 1000
for p_val in range(1, 5):
    for d_val in range(0, 3):
        
        ar_results_val, ar_results_test, ar_duration = AR_results(data_train = X_train, 
                                                                  data_val = X_val,
                                                                  data_test = X_test,
                                                                  p = p_val,
                                                                  d = d_val)
        if ar_results_val[1] < min_v:
            min_v = ar_results_val[1]
            print("\np:"+str(p_val)+" d:"+str(d_val))
            print("Duration: "+str(ar_duration))
            print("Validation RMSE:  ", ar_results_val[0])
            print("Validation NRMSE: ", ar_results_val[1])
            print("Test RMSE:  ", ar_results_test[0])
            print("Test NRMSE: ", ar_results_test[1])
  
        
print("\nAutoregression with Matrix Coefficients")
min_v = 1000
for p_val in range(1, 5):
    for d_val in range(0, 3):
        
        var_results_val, var_results_test, var_duration = VAR_results(data_train = X_train, 
                                                                      data_val = X_val, 
                                                                      data_test = X_test,
                                                                      p = p_val,
                                                                      d = d_val)
        if var_results_val[1] < min_v:
            min_v = var_results_val[1]
            print("\np:"+str(p_val)+" d:"+str(d_val))
            print("Duration: "+str(var_duration))
            print("Validation RMSE:  ", var_results_val[0])
            print("Validation NRMSE: ", var_results_val[1])
            print("Test RMSE:  ", var_results_test[0])
            print("Test NRMSE: ", var_results_test[1])




# print("Test RMSE:  ", ar_results_test[0])
# print("Test NRMSE: ", ar_results_test[1])
# print("Duration AR: ", ar_duration)



# print("Test RMSE:  ", var_results_test[0])
# print("Test NRMSE: ", var_results_test[1])
# print("Duration VAR: ", var_duration)

'''
def VAR_results2(data_train, data_val, data_test, p, d):
    
    if d>0:
        data_train_original = copy.deepcopy(data_train)
        data_train, inv = difference(data_train, d)
    
    start = time.clock()    
    model = VAR(data_train.T)
    results = model.fit(p)
    end = time.clock()
    duration = end - start
    #A = results.coefs
    
    predictions_val = results.forecast(data_train[..., -p:].T, data_val.shape[-1])
    
    if d>0:
        predictions_val = inv_difference(np.append(data_train, predictions_val.T, axis=-1), inv, d)
        
    rmse = compute_rmse(predictions_val[..., -data_val.shape[-1]:], data_val)
    nrmse = compute_nrmse(predictions_val[..., -data_val.shape[-1]:], data_val)
    results_val = [rmse, nrmse]
    
    # Testing Phase
    # data_test_start = np.append(data_train_original, data_val, axis=-1)
    # if d>0:
    #     data_test_start, inv = difference(data_test_start, d)
    
    # predictions_test = results.forecast(data_test_start[..., -p:].T, data_test.shape[-1])
    
    # if d>0:
    #     predictions_test = inv_difference(np.append(data_test_start, predictions_test.T, axis=-1), inv, d)
    
    # rmse = compute_rmse(predictions_test[..., -data_test.shape[-1]:], data_test)
    # nrmse = compute_nrmse(predictions_test[..., -data_test.shape[-1]:], data_test)
    results_test = []#[rmse, nrmse]
    
    return results_val, results_test, duration
'''



# var_results2_val, var_results2_test, var_duration2 = VAR_results2(data_train = X_train, 
#                                                                   data_val = X_val, 
#                                                                   data_test = X_test,
#                                                                   p = 2,
#                                                                   d = 1)

# print("\nAutoregression with Matrix Coefficients 2")
# print("Validation RMSE:  ", var_results2_val[0])
# print("Validation NRMSE: ", var_results2_val[1])
# print("Test RMSE:  ", var_results2_test[0])
# print("Test NRMSE: ", var_results2_test[1])
# print("Duration VAR: ", var_duration)


