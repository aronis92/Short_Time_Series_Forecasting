from functions.utils import create_synthetic_data
from functions.BHT_AR_functions import train
from functions.AR_functions import fit_ar
from functions.MAR_functions import fit_mar
from functions.MDT_functions import MDT
import matplotlib.pyplot as plt
import numpy as np

def plot_results(data, title):
    epoch = [i + 1 for i in range(len(data))]
    plt.figure(figsize = (12,5))
    plt.title(title)
    #plt.plot(epoch[1:], data[1:])
    plt.plot(epoch, data)

# Create/Load Dataset
np.random.seed(0)
X = np.load('data/traffic_40.npy').T
# X = create_synthetic_data(p = 2, dim = 5, n_samples=100)

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



