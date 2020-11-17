##################################################
##                                              ##
##  This file contains tests conducted for the  ##
##     Scalar Coefficients BHT_AR Algorithm     ##
##                                              ##
##################################################

from functions.utils import plot_results, get_data, get_ranks, difference
from functions.BHT_AR_functions import BHTAR, BHTAR_test, MDT
import matplotlib.pyplot as plt
import numpy as np
import time
np.random.seed(0)

#                             Index  Var x Time
datasets = ['macro', #__________0     12 x 203
            'elnino', #_________1     12 x 61
            'stackloss', #______2      4 x 21
            'nightvisitors', #__3      8 x 56
            'mortality', #______4      2 x 72
            'ozone', #__________5      8 x 203
            'inflation', #______6      8 x 123  
            'nasdaq', #_________7     82 x 40560 # Pending
            'yahoo', #__________8     5 x 2469  
            'book'] #___________9     3 x sum(Ns)

# Load the Dataset
data_name = datasets[0]
X_train, X_val, X_test = get_data(dataset = data_name,
                                  Ns = [50, 1, 1])

# Plot the loaded data
# plt.figure(figsize = (12,5))
# #plt.ylim(20, 70)
# plt.plot(X_train.T)
# plt.show()


# Set the algorithm's parameters
parameters = {'R1':2,
              'R2':2,
              'p': 2,
              'r': 2,
              'd': 2, 
              'lam': 1,
              'max_epoch': 15,
              'threshold': 0.000001}


file = open("results/BHT_AR_" + data_name + ".txt", 'a')

ds = [2, 3, 4, 5]

for p_val in range(1,6):
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

# l_list = [.1, .2, .3, .4, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# for l in l_list:
#     parameters['lam'] = l
    
                    start = time.clock()
                    convergences, changes, A, prediction, Us = BHTAR(data_train = X_train,
                                                                     data_val = X_val,
                                                                     par = parameters,
                                                                     mod = "AR")
                    end = time.clock()
                    duration_AR = end - start
                    
                    rmse_AR = changes[:,0]
                    nrmse_AR = changes[:,1]
                    
                    # Validation
                    #print("\nR1:", parameters['R1'], " R2:", parameters['R2'], " p:", parameters['p'], " r:", parameters['r'])
                    #print("\nlam:", parameters['lam'])
                    #print("Validation RMSE_AR: ", rmse_AR[-1], min(rmse_AR))
                    # print("Validation NRMSE_AR: ", nrmse_AR[-1], min(nrmse_AR))
                    #print("Validation duration_AR: ", duration_AR)
                    
                    if nrmse_AR[-1] < 0.01047:
                    #    file.write("\nlam:", parameters['lam']")
                        file.write("\nR1:"+str(R1_val)+" R2:"+str(R2_val)+" p:"+str(parameters['p'])+" r:"+str(r_val)+" d:"+str(d_val)) 
                        file.write("\nValidation RMSE_AR: "+str(rmse_AR[-1])) 
                        file.write("\nValidation NRMSE_AR: "+str(nrmse_AR[-1])+"\n") 
            

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
#                                    mod = "AR")

#print("Test RMSE_VAR: ", test_rmse)
# print("Test NRMSE_AR: ", test_nrmse)




# plot_results(convergences, 'BHT_AR Convergence', "Convergence Value")
# plot_results(changes[:,0], 'BHT_AR RMSE', "RMSE Value")
# plot_results(changes[:,1], 'BHT_AR ΝRMSE', "NRMSE Value")






# Test
# rmse_AR, nrmse_AR = predict("AR", Us, A, parameters, X, X_test)

# print("Test RMSE_AR: ", rmse_AR)
# print("Test NRMSE_AR: ", nrmse_AR)







