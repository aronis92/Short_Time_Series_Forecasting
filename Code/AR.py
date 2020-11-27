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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(0)

#                             Index  Var x Time
datasets = ['macro', #__________0     12 x 203    # DONE
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
                                  Ns = [20, 7, 7])

# Plot the loaded data
# plt.figure(figsize = (12,5))
# #plt.ylim(20, 70)
# plt.plot(X_train.T)
# plt.show()


# Set the algorithm's parameters
parameters = {'R1': 3,
              'R2': 4,
              'p': 1,
              'r': 5,
              'd': 2,
              'lam': 1,
              'max_epoch': 15,
              'threshold': 0.000001}

   
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
print("\nR1:", parameters['R1'], " R2:", parameters['R2'], " p:", parameters['p'], " r:", parameters['r'], " d:", parameters['d'], " lam:", parameters['lam'])
print("Validation RMSE_AR: ", rmse_AR[-1])#, min(rmse_AR))
print("Validation NRMSE_AR: ", nrmse_AR[-1])#, min(nrmse_AR))
print("Validation duration_AR: ", duration_AR)
                    



''' Check X_test_start'''
X_test_start = np.append(X_train, X_val, axis=-1)

test_rmse, test_nrmse = BHTAR_test(X_test_start,
                                   X_test,
                                   A, 
                                   Us,
                                   parameters, 
                                   mod = "AR")

print("Test RMSE_AR: ", test_rmse)
print("Test NRMSE_AR: ", test_nrmse)




plot_results(convergences, 'BHT_AR Convergence', "Convergence Value")
#plot_results(changes[:,0], 'BHT_AR RMSE', "RMSE Value")
plot_results(changes[:,1], 'BHT_AR ΝRMSE', "NRMSE Value")