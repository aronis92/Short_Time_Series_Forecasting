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
            'stackloss', #______8     4 x 21      #
            'book1'] #__________9     3 x sum(Ns) #

# Load the Dataset
data_name = datasets[9]
X_train, X_val, X_test = get_data(dataset = data_name,
                                  Ns = [20, 1, 1])

# Set the algorithm's parameters
parameters = {'R1':2,
              'R2':2,
              'p': 1,
              'r': 6,
              'd': 3,
              'lam': 1,
              'max_epoch': 15,
              'threshold': 0.000001}


file = open("results/BHT_VAR_" + data_name + ".txt", 'a')

ds = [0, 1, 2]#, 3, 4, 5]
min_v = 1000
for p_val in range(1, 2):#,6):
    parameters['p'] = p_val
    
    for r_val in range(2, 11):
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

                    start = time.clock()
                    convergences, changes, A, prediction, Us = BHTAR(data_train = X_train,
                                                                     data_val = X_val,
                                                                     par = parameters,
                                                                     mod = "VAR")
                    end = time.clock()
                    duration_VAR = end - start
                    
                    rmse_VAR = changes[:,0]
                    nrmse_VAR = changes[:,1]
                    
                    if nrmse_VAR[-1] < min_v:
                        min_v = nrmse_VAR[-1]
                        file.write("\nR1:"+str(parameters['R1'])+"  R2:"+str(parameters['R2'])+"  p:"+str(parameters['p'])+"  r:"+str(parameters['r'])+"  d:"+str(parameters['d'])+"  lam:" + str(parameters['lam'])) 
                        file.write("\nValidation RMSE_VAR: "+str(rmse_VAR[-1])) 
                        file.write("\nValidation NRMSE_VAR: "+str(nrmse_VAR[-1])+"\n")
file.close()


'''Lambda Grid Search'''
# parameters = {'R1':2,
#               'R2':4,
#               'p': 1,
#               'r': 8,
#               'd': 2,
#               'lam': 1,
#               'max_epoch': 15,
#               'threshold': 0.000001}

# file = open("results/BHT_VAR_" + data_name + "_v2.txt", 'a')
# l_list = [.1, .2, .3, .4, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 50, 100, 1000]
# min_v = 1000
# for l in l_list:
#     parameters['lam'] = l
#     start = time.clock()
#     convergences, changes, A, prediction, Us = BHTAR(data_train = X_train,
#                                                      data_val = X_val,
#                                                      par = parameters,
#                                                      mod = "VAR")
#     end = time.clock()
#     duration_VAR = end - start
    
#     rmse_VAR = changes[:,0]
#     nrmse_VAR = changes[:,1]
    
#     if nrmse_VAR[-1] < min_v:
#         min_v = nrmse_VAR[-1]
#         file.write("\nlambda: "+str(l))
#         file.write("\nValidation RMSE_VAR: "+str(rmse_VAR[-1])) 
#         file.write("\nValidation NRMSE_VAR: "+str(nrmse_VAR[-1])+"\n")

# file.close()



