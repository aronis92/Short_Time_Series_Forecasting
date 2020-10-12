########################################################
##                                                    ##
##  This file contains main execution of the BHT_VAR  ##
##                                                    ##
########################################################

from functions.utils import create_synthetic_data, create_synthetic_data2, plot_results
from functions.BHT_AR_functions import train, predict
import numpy as np
import pandas as pd
import time
#from functions.AR_functions import fit_ar
#from functions.MAR_functions import fit_mar
#from functions.MDT_functions import MDT


'''Create/Load Dataset'''
np.random.seed(0)
# X = create_synthetic_data2(p = 2, dim = 10, n_samples=11)
# X = create_synthetic_data(p = 2, dim = 100, n_samples=41)
X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = 100)
X = X.to_numpy()
X = X.T
X2 = X[:, -41:]
X_test = X2[:, -1:]
X2 = X2[:, :-1]


'''~~~~~~~~~~~~~~~~~~~'''
'''      BHT_VAR      '''
'''~~~~~~~~~~~~~~~~~~~'''

# Set the parameters for BHT_VAR
parameters = {'p': 3,
              'r': 3,
              'lam': 2.1, #2.1 for VAR
              'max_epoch': 15,
              'threshold': 0.000001}

print("\nBHT_VAR\np:", parameters['p'], " r:", parameters['r'])

start = time.clock()
Us, convergences, changes, A, prediction = train(data = X,
                                                 par = parameters,
                                                 model = "VAR")
end = time.clock()
duration_VAR = end - start

rmse_VAR = changes[:,0]
nrmse_VAR = changes[:,1]
print("RMSE_VAR: ", min(rmse_VAR))
print("NRMSE_VAR: ", min(nrmse_VAR))

# plot_results(convergences, 'BHT_VAR Convergence', "Convergence Value")
# plot_results(changes[:,0], 'BHT_VAR RMSE', "RMSE Value")
# plot_results(changes[:,1], 'BHT_VAR ΝRMSE', "NRMSE Value")