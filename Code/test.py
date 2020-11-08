import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_results(data, title, ytitle):
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.xlabel('Dataset Volume')
    plt.ylabel(ytitle)
    plt.xticks(ticks=[0, 1, 2, 3, 4],
               labels=['100 points', '40 points', '20 points', '10 points', '5 points'])
    
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    plt.plot(data[:,0], label="AR", marker='o')
    plt.plot(data[:,1], label="VAR", marker='o')
    plt.plot(data[:,2], label="Prophet", marker='o')
    plt.plot(data[:,3], label="BHT_AR", marker='o')
    plt.plot(data[:,4], label="BHT_VAR", marker='o')
    
    plt.legend(fancybox = True,
               framealpha = 1,
               shadow = True,
               borderpad = 1,
               ncol = 5,
               fontsize = 12)
    
    
#              AR________VAR_______Prophet___BHT-AR____BHT-VAR_
X = np.array([[0.071759, 0.025453, 0.091920, 0.068464, 0.087860],  # 100
              [0.070842, 0.125955, 0.186396, 0.068168, 0.066181],  #  40
              [0.080897, 0.129356, 0.149310, 0.081648, 0.076533],  #  20
              [0.130220, 0.152955, 0.104766, 0.069972, 0.059880],  #  10
              [0.226482, 0.221791, 0.161345, 0.173531, 0.146240]]) #   5

plot_results(X, "Synthetic NRMSE over Volume", "NRMSE")


#              AR________VAR_______Prophet___BHT-AR____BHT-VAR_
X = np.array([[0.000587, 0.003934, 0.005801, 0.000446, 0.000588],  # 100
              [0.000770, 0.001100, 0.001152, 0.000708, 0.000643],  #  40
              [0.001935, 0.002832, 0.001437, 0.000783, 0.000710],  #  20
              [0.001935, 0.002832, 0.001484, 0.001183, 0.001174],  #  10
              [0.002495, 0.001836, 0.004079, 0.001667, 0.001534]]) #   5

plot_results(X, "NASDAQ-100 NRMSE over Volume", "NRMSE")




import pandas as pd
import numpy as np
from fbprophet import Prophet
from statsmodels.tsa.arima_process import arma_generate_sample
import time

def compute_rmse(y_pred, y_true):
    rmse = np.sqrt( np.linalg.norm(y_pred - y_true)**2 / np.size(y_true) )
    return rmse

def compute_nrmse(y_pred, y_true):
    # t1 = np.linalg.norm(y_pred - y_true)**2 / np.size(y_true)
    t1 = compute_rmse(y_pred, y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    nrmse = t1 / t2
    return nrmse

def book_data(sample_size):
    np.random.seed(69)
    A1 = np.array([[.3, -.2, .04], [-.11, .26, -.05], [.08, -.39, .39]])
    A2 = np.array([[.28, -.08, .07], [-.04, .36, -.1], [-.33, .05, .38]])
    total = sample_size + 2000
    X_total = np.zeros((3, total))
    X_total[..., 0:2] = np.random.rand(3,2)
    for i in range(2, total):
        X_total[..., i] = np.dot(-A1, X_total[..., i-1]) + np.dot(-A2, X_total[..., i-2]) + np.random.rand(3,)
    return X_total[..., (total-sample_size):], A1, A2

n_train = 40
n_val = 5
n_test = 5
n_total = n_train + n_val + n_test

#data = pd.read_csv('/content/nasdaq100_padding.csv',  nrows = n_total)
#data = data.to_numpy()
#data = data.T
X, _, _ = book_data(sample_size=n_total)

X_train = np.append(X[..., :n_train], X[..., n_train:(n_val+n_train)], axis=-1)
X_train = X_train[..., n_val:]
X_test = X[..., -n_test:]

#import matplotlib.pyplot as plt
#plt.figure(figsize=(12,5))
#plt.ylim(-1, 2)
#plt.plot(X_train.T)
#plt.show()

X_train = pd.DataFrame(X_train.T)
#print(X_train.header())
ds = pd.date_range('2015-02-24', periods = n_train, freq='D')
ds = pd.DataFrame(ds.date)
ds = ds.rename(columns={0:'ds'})
new_data = pd.concat([ds, X_train], axis = 1)
print(new_data.tail())
print(new_data.shape)




