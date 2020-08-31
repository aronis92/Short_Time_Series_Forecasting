#######################################################
##                                                   ##
##  This file contains main execution of the BHT_AR  ##
##                                                   ##
#######################################################

from functions.utils import create_synthetic_data, plot_results
from functions.BHT_AR_functions import train
import numpy as np
#from functions.AR_functions import fit_ar
#from functions.MAR_functions import fit_mar
#from functions.MDT_functions import MDT


# Create/Load Dataset
np.random.seed(0)
X = create_synthetic_data(p = 2, dim = 5, n_samples=100)

# X = np.load('data/traffic_40.npy').T
# plt.figure(figsize=(13,5))
# plt.plot(X.T)

# Set some parameters
parameters = {'r': 5,
              'p': 2,
              # 'ranks': [0, 0],
              'lam': 1,
              'max_epoch': 10,
              'threshold': 0.001}

Us, convergences, changes, A, prediction = train(data = X,
                                                 par = parameters,
                                                 model = "VAR")


plot_results(convergences, 'Conv')
# changes[:, 0] = changes[:, 0] * 100 - 20
# changes[:, 1] = changes[:, 1] * 10**4 - 34
plot_results(changes, 'RMSE/NRMSE')



