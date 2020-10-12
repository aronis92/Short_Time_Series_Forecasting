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
# X = create_synthetic_data2(p = 2, dim = 10, n_samples=11)
# X = create_synthetic_data(p = 2, dim = 100, n_samples=41)
X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = 100)
X = X.to_numpy()
X = X.T
X2 = X[:, -41:]
X_test = X2[:, -1:]
X2 = X2[:, :-1]

# X = np.load('data/traffic_40.npy').T
# import matplotlib.pyplot as plt
# plt.figure(figsize=(13,5))
# plt.plot(X.T)


'''~~~~~~~~~~~~~~~~~~~~'''
'''       BHT_AR       '''
'''~~~~~~~~~~~~~~~~~~~~'''

# Set the parameters for BHT_AR
parameters = {'p': 2,
              'r': 5, #8,
              'lam': 0.1, #5,
              'max_epoch': 15,
              'threshold': 0.000001}

start = time.clock()
Us, convergences, changes, A, prediction = train(data = X,
                                                  par = parameters,
                                                  mod = "AR")
end = time.clock()
duration_AR = end - start

print("BHT_AR\np:", parameters['p'], " r:", parameters['r'])

# Validation
rmse_AR = changes[:,0]
nrmse_AR = changes[:,1]
print("Validation RMSE_AR: ", min(rmse_AR))
print("Validation  RMSE_AR: ", min(nrmse_AR))

#Test
rmse_AR, nrmse_AR = predict("AR", Us, A, parameters, X, X_test)

print("RMSE_AR: ", min(rmse_AR))
print("NRMSE_AR: ", min(nrmse_AR))

# plot_results(convergences, 'BHT_AR Convergence', "Convergence Value")
# plot_results(changes[:,1], 'BHT_AR ΝRMSE', "NRMSE Value")
# plot_results(changes[:,0], 'BHT_AR RMSE', "RMSE Value")

