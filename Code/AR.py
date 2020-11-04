#######################################################
##                                                   ##
##  This file contains main execution of the BHT_AR  ##
##                                                   ##
#######################################################

from functions.utils import get_matrix_coeff_data, create_synthetic_data2, plot_results, book_data
from functions.BHT_AR_functions_test import BHTAR, BHTAR_test
import numpy as np
import pandas as pd
import time
#from functions.AR_functions import fit_ar
#from functions.MAR_functions import fit_mar
#from functions.MDT_functions import MDT


'''Create/Load Dataset'''
np.random.seed(0)
n_train = 100
n_val = 2
n_test = 5
n_total = n_train + n_val + n_test

# X = create_synthetic_data2(p = 2, dim = 10, n_samples=6)
# X = create_synthetic_data(p = 2, dim = 100, n_samples=41)
X, _, _ = get_matrix_coeff_data(sample_size=n_total, n_rows=6, n_columns=5)
# X, _, _ = book_data(sample_size=1001)
# X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = 6)
# X = X.to_numpy()
# X = X.T
# X2 = X[:, -41:]
# X_test = X2[:, -1:]
# X2 = X2[:, :-1]
# X = X[:, :41]

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''       BHT_AR_Scalar_Coefficients       '''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# Set the parameters for BHT_AR
parameters = {'R1':5,
              'R2':3,
              'p': 2,
              'r': 5, #8,
              'lam': 1, #5,
              'max_epoch': 15,
              'threshold': 0.000001}

data_train = X[..., :n_train]
data_val = X[..., n_train:(n_val+n_train)]
data_test = X[..., -n_test:]

#start = time.clock()
convergences, changes, A, prediction, Us = BHTAR(X_train = data_train,
                                                 X_val = data_val,
                                                 par = parameters,
                                                 mod = "AR")
#end = time.clock()
#duration_AR = end - start

if data_val.shape[-1] >= parameters['p'] + parameters['r'] - 1:
    dummy = data_val 
else:
    dummy = np.append(data_train[..., -(parameters['p'] + parameters['r'] - 1 - data_val.shape[-1]):], data_val, axis=-1)

test_rmse, test_nrmse = BHTAR_test(dummy,
                                   data_test,
                                   A, 
                                   Us,
                                   parameters, 
                                   mod = "VAR")

print("BHT_AR\nR1:", parameters['R1'], " R2:", parameters['R2'], " p:", parameters['p'], " r:", parameters['r'])

# Validation
rmse_AR = changes[:,0]
nrmse_AR = changes[:,1]
#print("Validation RMSE_AR: ", min(rmse_AR))
print("Validation NRMSE_AR: ", nrmse_AR[-1])#min(nrmse_AR))
#print("Validation duration_AR: ", duration_AR)

#print("Test RMSE_VAR: ", test_rmse)
print("Test NRMSE_VAR: ", test_nrmse)

# plot_results(convergences, 'BHT_AR Convergence', "Convergence Value")
# plot_results(changes[:,0], 'BHT_AR RMSE', "RMSE Value")
#plot_results(changes[:,1], 'BHT_AR ΝRMSE', "NRMSE Value")






# Test
# rmse_AR, nrmse_AR = predict("AR", Us, A, parameters, X, X_test)

# print("Test RMSE_AR: ", rmse_AR)
# print("Test NRMSE_AR: ", nrmse_AR)




