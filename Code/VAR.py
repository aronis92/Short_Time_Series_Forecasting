##################################################
##                                              ##
##  This file contains tests conducted for the  ##
##     Matrix Coefficients BHT_AR Algorithm     ##
##                                              ##
##################################################

from functions.utils import plot_results, get_data, get_ranks, difference
from functions.BHT_AR_functions import BHTAR, BHTAR_test, MDT
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(0)

#                             Index  Var x Time
datasets = ['macro', #__________0     12 x 203    # 
            'elnino', #_________1     12 x 61     # DONE
            'ozone', #__________2      8 x 203    # DONE
            'nightvisitors', #__3      8 x 56     #
            'inflation', #______4      8 x 123    # DROP PROBABLY
            'nasdaq', #_________5     82 x 40560  # Pending
            'yahoo', #__________6     5 x 2469    #
            'book', #___________7     3 x sum(Ns) #
            'stackloss'] #______8     4 x 21   

# Load the Dataset
data_name = datasets[8]
X_train, X_val, X_test = get_data(dataset = data_name,
                                  Ns = [19, 1, 1])

# Plot the loaded data
# plt.figure(figsize = (12,5))
# #plt.ylim(20, 70)
# plt.plot(X_train.T)
# plt.show()


# Set the algorithm's parameters
parameters = {'R1':3,
              'R2':3,
              'p': 1,
              'r': 4,
              'd': 2,
              'lam': 1,
              'max_epoch': 15,
              'threshold': 0.000001}


file = open("results/BHT_VAR_" + data_name + ".txt", 'a')

ds = [0, 1, 2, 3, 4, 5]
l_list = [.1, .2, .3, .4, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 50, 100, 1000]
min_v = 1000
for p_val in range(1,6):
    parameters['p'] = p_val
    
    for r_val in range(2, 6):#11):
        parameters['r'] = r_val
        
        for d_val in ds:
            parameters['d'] = d_val
            print("p:", p_val, "r:", r_val, "d:", d_val)
            X_hat, _ = MDT(X_train, parameters['r'])
            if parameters['d']>0:
                X_hat, _ = difference(X_hat, parameters['d'])
            Rs = get_ranks(X_hat)
            
            for R1_val in range(2, Rs[0]+1):
                parameters['R1'] = R1_val
                
                for R2_val in range(2, r_val+1):
                    parameters['R2'] = R2_val

# for l in l_list:
#     parameters['lam'] = l
        
                    start = time.clock()
                    convergences, changes, A, prediction, Us = BHTAR(data_train = X_train,
                                                                     data_val = X_val,
                                                                     par = parameters,
                                                                     mod = "VAR")
                    end = time.clock()
                    duration_VAR = end - start
                    
                    rmse_VAR = changes[:,0]
                    nrmse_VAR = changes[:,1]
                    
                    # Validation
                    #print("\nR1:", parameters['R1'], " R2:", parameters['R2'], " p:", parameters['p'], " r:", parameters['r'])
                    #print("\nlam:", parameters['lam'])
                    # print("Validation RMSE_VAR: ", rmse_VAR[-1], min(rmse_VAR))
                    # print("Validation NRMSE_VAR: ", nrmse_VAR[-1], min(nrmse_VAR))
                    #print("Validation duration_VAR: ", duration_VAR)
                    
                    if nrmse_VAR[-1] < min_v:
                        min_v = nrmse_VAR[-1]
                        #file.write("\nlambda: "+str(l))
                        file.write("\nR1:"+str(R1_val)+"  R2:"+str(R2_val)+"  p:"+str(parameters['p'])+"  r:"+str(r_val)+"  d:"+str(d_val)+"  lam:" + str(parameters['lam'])) 
                        file.write("\nValidation RMSE_VAR: "+str(rmse_VAR[-1])) 
                        file.write("\nValidation NRMSE_VAR: "+str(nrmse_VAR[-1])+"\n")
            

file.close()



''' Check X_test_start'''
# Prepare the data needed for the testing predictions
# if X_val.shape[-1] >= parameters['p'] + parameters['r'] - 1:
#     X_test_start = X_val 
# else:
#     X_test_start = np.append(X_train[..., -(parameters['p'] + parameters['r'] - 1 - X_val.shape[-1]):], X_val, axis=-1)


# test_rmse, test_nrmse = BHTAR_test(X_test_start,
#                                    X_test,
#                                    A, 
#                                    Us,
#                                    parameters, 
#                                    mod = "VAR")

#print("Test RMSE_VAR: ", test_rmse)
# print("Test NRMSE_VAR: ", test_nrmse)




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



