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
# X = create_synthetic_data2(p = 2, dim = 10, n_samples=100)
# X = create_synthetic_data(p = 2, dim = 40, n_samples=100)
X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = 40)
X = X.to_numpy()
X = X.T

# X = np.load('data/traffic_40.npy').T
# plt.figure(figsize=(13,5))
# plt.plot(X.T)

# Set some parameters
parameters = {'r': 2,
              'p': 3,
              # 'ranks': [0, 0],
              'lam': 5,
              'max_epoch': 10,
              'threshold': 0.000001}

Us, convergences, changes, A, prediction = train(data = X,
                                                 par = parameters,
                                                 model = "VAR")


# plot_results(convergences, 'Conv')
# # changes[:, 0] = changes[:, 0] * 100 - 20
# # changes[:, 1] = changes[:, 1] * 10**4 - 34
# plot_results(changes, 'RMSE/NRMSE')

rmse_VAR = changes[:,0]
nrmse_VAR = changes[:,1]

Us, convergences, changes, A, prediction = train(data = X,
                                                 par = parameters,
                                                 model = "AR")
rmse_AR = changes[:,0]
nrmse_AR = changes[:,1] 

print("RMSE_AR: ", rmse_AR[-1])
print("RMSE_VAR: ", rmse_VAR[-1])

# del A, Us, changes, convergences, parameters, X, prediction