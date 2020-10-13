#######################################################
##                                                   ##
##  This file contains main execution of the BHT_AR  ##
##                                                   ##
#######################################################

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
# X = create_synthetic_data2(p = 2, dim = 10, n_samples=6)
# X = create_synthetic_data(p = 2, dim = 100, n_samples=41)
X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = 6)
X = X.to_numpy()
X = X.T
# X2 = X[:, -41:]
# X_test = X2[:, -1:]
# X2 = X2[:, :-1]
# X = X[:, :41]

'''~~~~~~~~~~~~~~~~~~~~'''
'''       BHT_AR       '''
'''~~~~~~~~~~~~~~~~~~~~'''

# Set the parameters for BHT_AR
parameters = {'R1':3,
              'R2':2,
              'p': 2,
              'r': 2, #8,
              'lam': 2, #5,
              'max_epoch': 15,
              'threshold': 0.000001}

start = time.clock()
Us, convergences, changes, A, prediction = train(data = X,
                                                  par = parameters,
                                                  mod = "AR")
end = time.clock()
duration_AR = end - start

print("BHT_AR\nR1:", parameters['R1'], " R2:", parameters['R2'], " p:", parameters['p'], " r:", parameters['r'])

# Validation
rmse_AR = changes[:,0]
nrmse_AR = changes[:,1]
print("Validation RMSE_AR: ", min(rmse_AR))
print("Validation  RMSE_AR: ", min(nrmse_AR))
print("Validation duration_AR: ", duration_AR)

# plot_results(convergences, 'BHT_AR Convergence', "Convergence Value")
# plot_results(changes[:,0], 'BHT_AR RMSE', "RMSE Value")
# plot_results(changes[:,1], 'BHT_AR ΝRMSE', "NRMSE Value")

# Test
# rmse_AR, nrmse_AR = predict("AR", Us, A, parameters, X, X_test)

# print("Test RMSE_AR: ", rmse_AR)
# print("Test NRMSE_AR: ", nrmse_AR)







