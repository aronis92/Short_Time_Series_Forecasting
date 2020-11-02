########################################################
##                                                    ##
##  This file contains main execution of the BHT_VAR  ##
##                                                    ##
########################################################

from functions.utils import get_matrix_coeff_data, create_synthetic_data2, plot_results, book_data
from functions.BHT_AR_functions import train, predict
import numpy as np
import pandas as pd
import time
#from functions.AR_functions import fit_ar
#from functions.MAR_functions import fit_mar
#from functions.MDT_functions import MDT


'''Create/Load Dataset'''
np.random.seed(0)
# X = create_synthetic_data2(p = 2, dim = 10, n_samples=6)
# X = create_synthetic_data(p = 2, dim = 100, n_samples=41)
X, _, _ = get_matrix_coeff_data(sample_size=201, n_rows=3, n_columns=3)
# X, _, _ = book_data(sample_size=1001)
# X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = 6)
# X = X.to_numpy()
# X = X.T
# X2 = X[:, -41:]
# X_test = X2[:, -1:]
# X2 = X2[:, :-1]


'''~~~~~~~~~~~~~~~~~~~'''
'''      BHT_VAR      '''
'''~~~~~~~~~~~~~~~~~~~'''

# Set the parameters for BHT_VAR
parameters = {'R1':5,
              'R2':3,
              'p': 2,
              'r': 5,
              'lam': 1, #2.1 for VAR
              'max_epoch': 15,
              'threshold': 0.000001}


#start = time.clock()
Us, convergences, changes, A, prediction = train(data = X,
                                                 par = parameters,
                                                 mod = "VAR")
#end = time.clock()
#duration_VAR = end - start

print("\nBHT_VAR\nR1:", parameters['R1'], " R2:", parameters['R2'], " p:", parameters['p'], " r:", parameters['r'])

# Validation
rmse_VAR = changes[:,0]
nrmse_VAR = changes[:,1]
#print("RMSE_VAR: ", min(rmse_VAR))
print("NRMSE_VAR: ", nrmse_VAR[-1])#min(nrmse_VAR))
#print("Validation duration_VAR: ", duration_VAR)

#plot_results(convergences, 'BHT_VAR Convergence', "Convergence Value")
#plot_results(changes[:,0], 'BHT_VAR RMSE', "RMSE Value")
#plot_results(changes[:,1], 'BHT_VAR ΝRMSE', "NRMSE Value")

# Test
# rmse_VAR, nrmse_VAR = predict("VAR", Us, A, parameters, X, X_test)

# print("Test RMSE_AR: ", rmse_VAR)
# print("Test NRMSE_AR: ", nrmse_VAR)


# kron_T = np.kron(Us[1].T, Us[0].T)
# kron_2 = np.kron(Us[1], Us[0])
# A1_restored = np.linalg.multi_dot([np.linalg.pinv(kron_T), A[1], np.linalg.pinv(kron_2)])
# A2_restored = np.linalg.multi_dot([np.linalg.pinv(kron_T), A[2], np.linalg.pinv(kron_2)])

# from numpy import linalg as la
# import matplotlib.pyplot as plt

# np.random.seed(69)
# total = 2000
# X_total = np.zeros((15, total))
# X_total[..., 0:2] = np.random.rand(15, 2)
# for i in range(2, total):
#     X_total[..., i] = np.dot(A1, X_total[..., i-1]) + np.dot(A2, X_total[..., i-2]) + np.random.rand(15,)
# X = X_total[..., (total-50):]
# plt.figure(figsize = (12,5))
# X_vectorized = X[:5, :]
# plt.plot(X_vectorized.T)
# plt.show()


# X_total = np.zeros((9, total))
# X_total[..., 0:2] = np.random.rand(9, 2)
# for i in range(2, total):
#     X_total[..., i] = np.dot(A[0], X_total[..., i-1]) + np.dot(A[1], X_total[..., i-2]) + np.random.rand(9,)
# X = X_total[..., (total-50):]
# plt.figure(figsize = (12,5))
# X_vectorized = X[:3, :]
# plt.plot(X_vectorized.T)
# plt.show()



