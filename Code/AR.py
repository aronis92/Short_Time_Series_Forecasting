#######################################################
##                                                   ##
##  This file contains main execution of the BHT_AR  ##
##                                                   ##
#######################################################

from functions.utils import create_synthetic_data, create_synthetic_data2, plot_results
from functions.BHT_AR_functions import train
import numpy as np
import pandas as pd
#from functions.AR_functions import fit_ar
#from functions.MAR_functions import fit_mar
#from functions.MDT_functions import MDT


# Create/Load Dataset
np.random.seed(0)
# X = create_synthetic_data2(p = 2, dim = 100, n_samples=40)
# X = create_synthetic_data(p = 2, dim = 100, n_samples=40)
X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = 41)
X = X.to_numpy()
X = X.T

# X = np.load('data/traffic_40.npy').T
# plt.figure(figsize=(13,5))
# plt.plot(X.T)

# Set some parameters
parameters = {'r': 5,
              'p': 2,
              # 'ranks': [0, 0],
              'lam': 2, #2.1 for VAR
              'max_epoch': 15,
              'threshold': 0.000001}

print("p:", parameters['p'], " r:", parameters['r'])
Us, convergences, changes, A, prediction = train(data = X,
                                                 par = parameters,
                                                 model = "AR")
# plot_results(convergences, 'BHT_AR Convergence')
# plot_results(changes[:,1], 'BHT_AR ΝRMSE')
# plot_results(changes[:,0], 'BHT_AR RMSE')

rmse_AR = changes[:,0]
nrmse_AR = changes[:,1]


Us, convergences, changes, A, prediction = train(data = X,
                                                 par = parameters,
                                                 model = "VAR")
# plot_results(convergences, 'BHT_VAR Convergence')
# plot_results(changes[:,1], 'BHT_VAR ΝRMSE')
# plot_results(changes[:,0], 'BHT_VAR RMSE')

rmse_VAR = changes[:,0]
nrmse_VAR = changes[:,1]



print("RMSE_AR: ", min(rmse_AR))
print("RMSE_VAR: ", min(rmse_VAR))
print("NRMSE_AR: ", min(nrmse_AR))
print("NRMSE_VAR: ", min(nrmse_VAR))

# del A, Us, changes, convergences, parameters, X, prediction