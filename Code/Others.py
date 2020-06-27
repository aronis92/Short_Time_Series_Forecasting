# from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from functions.utils import my_data2, create_synthetic_data, compute_rmse, compute_nrmse
import numpy as np
import time, copy
from functions.ARIMA_functions import fit_ar, autocorrelation
import scipy as sp
import matplotlib.pyplot as plt

# X = np.load('input/traffic_40.npy').T
# X = my_data2(p = 2, dim = 5, n_samples = 100)
# X = create_synthetic_data(p=2, dim=5, n_samples=100)
# plt.figure(figsize=(13,5))
# plt.plot(X.T)

def VAR_results(data, p):
    start = time.clock()
    model = VAR(data[..., :-1].T)
    results = model.fit(2)

    A = results.coefs
    end = time.clock()
    prediction = results.forecast(data[..., -p:].T, 1)

    duration = end - start
    rmse = compute_rmse(prediction.T, data[..., -1])
    nrmse = compute_nrmse(prediction.T, data[..., -1])
    return A, duration, rmse, nrmse

def ARIMA_results(data, p, d, q):
    prediction = np.zeros([1, data.shape[0]])
    start = time.clock()
    alpha = 0
    for i in range(data.shape[0]):
        model = ARIMA(data[i, :-1].T, order=(p, d, q))
        res = model.fit()
        prediction[0, i] = res.forecast()
        alpha += res.polynomial_ar
    end = time.clock()

    duration = end - start
    rmse = compute_rmse(prediction.T, data[..., -1])
    nrmse = compute_nrmse(prediction.T, data[..., -1])
    return prediction.T, duration, rmse, nrmse, -alpha[1:]/data.shape[0] #res.polynomial_ar
    
# alpha_AR = fit_ar(X, p=2)
# # var_A, var_duration, var_rmse, var_nrmse = VAR_results(data=X, p=2)
# pred, arima_duration, arima_rmse, arima_nrmse, arima_A = ARIMA_results(data=X, p=2, d=0, q=0)

# del arima_duration, arima_rmse, arima_nrmse
# model = VARMAX(X[..., :-1], order = (2, 0), enforce_stationarity=False) # 
# model_fit = model.fit(maxiter = 100, disp = False)



