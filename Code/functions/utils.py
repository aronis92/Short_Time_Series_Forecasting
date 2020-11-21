####################################################
##                                                ##
##  This file contains various utility functions  ##
##                                                ##
####################################################

from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from random import random, seed
from numpy import linalg as la
import statsmodels.api as sm
from numpy import log
import tensorly as tl
import pandas as pd
import numpy as np
import copy


def plot_results(data, title, ytitle):
    """
    Plots the results
    
    Input:
        data: The data to plot
        title: The title of the graph
    """
    # plt.plot(epoch[1:], data[1:])
    epoch = [int(i + 1) for i in range(len(data))]
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ytitle)
    # plt.ylim(0.001924, 0.00193) # AR NRMSE
    # plt.ylim(0.2691, 0.2702) # AR RMSE
    # plt.ylim(0.001924, 0.00193) # VAR NRMSE
    # plt.ylim(0.089, 0.09) # VAR RMSE
    plt.plot(epoch, data)


def get_data(dataset, Ns):
    """
    Splits and loads the dataset's partitions
    
    Input:
        dataset: The dataset choice. Choose from "book", ""
        Ns: The train, validation, test split numbers in a list
        
    Returns:
        X_train: The training partition
        X_val: The validation partition
        X_test: The testing partition
    """
    np.random.seed(0)
    if dataset == "book":
        """
        Variables : 3
        Timepoints: sum(Ns)
        """
        X = book_data(sum(Ns))
    
    elif dataset == "book1":
        """
        Variables : 3
        Timepoints: sum(Ns)
        """
        X = book_data_1(sum(Ns))
        
    elif dataset == "nasdaq":
        """
        Variables : 82
        Timepoints: 40560
        """
        X = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv')
        X = X.to_numpy()
        X = X.T
    
    elif dataset == "inflation":
        """
        Variables : 8
        Timepoints: 123
        """
        filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
        df = pd.read_csv(filepath, parse_dates=['date'], index_col='date',  nrows = sum(Ns))
        X = df.to_numpy()
        X = X.T
    
    elif dataset == "yahoo":
        """
        Variables : 5
        Timepoints: 2469
        """
        filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/yahoo.csv'
        df = pd.read_csv(filepath, parse_dates=['date'], index_col='date', skiprows=range(1, 1400), nrows = sum(Ns)+100)
        df = df[['VIX.Open', 'VIX.High', 'VIX.Low', 'VIX.Close', 'VIX.Adjusted']]
        X = df.to_numpy()
        X = X.T
    
    elif dataset == "noise":
        X = np.random.normal(0, 1, (3, sum(Ns)))
    
    # elif dataset == "traffic":
    #     """
    #     Variables : 40
    #     Timepoints: 228
    #     """
    #     X = np.load('data/traffic_40.npy').T
        
    elif dataset == "macro":
        """
        Variables : 12
        Timepoints: 203
        """
        data = sm.datasets.macrodata.load_pandas().data
        X = data[['realgdp', 'realcons', 'realinv', 'realgovt', 'realdpi', 'cpi', 'm1', 'tbilrate', 'unemp', 'pop', 'infl', 'realint']]
        X = X.to_numpy()
        X = X.T
        
    elif dataset == "elnino":
        """
        Variables : 12
        Timepoints: 61
        """
        data = sm.datasets.elnino.load_pandas().data
        X = data.to_numpy()
        X = X[..., 1:].T
        
    # elif dataset == "copper":
    #     """
    #     Variables : 5
    #     Timepoints: 25
    #     """
    #     data = sm.datasets.copper.load_pandas().data
    #     X = data.to_numpy()
    #     X = X[..., :-1].T
        
    # elif dataset == "fertility":
    #     """
    #     Variables : 192
    #     Timepoints: 52
    #     """
    #     data = sm.datasets.fertility.load_pandas().data
    #     data = data.iloc[:, 4:-2]
    #     data = data.dropna()
    #     X = data.to_numpy()
        
    elif dataset == "stackloss":
        """
        Variables : 4
        Timepoints: 21
        """
        data = sm.datasets.stackloss.load_pandas().data
        X = data.to_numpy()
        X = X.T
        
    elif dataset == "nightvisitors":
        """
        Variables : 8
        Timepoints: 56
        """
        filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/nightvisitors_multi_ts.csv'
        df = pd.read_csv(filepath, nrows = 100)
        X = df.to_numpy()
        X = X.T
        
    elif dataset == "mortality":
        """
        Variables : 2
        Timepoints: 72
        """
        filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/mortality.csv'
        df = pd.read_csv(filepath, nrows = sum(Ns))
        X = df.to_numpy()
        X = X[..., 1:].T  
        
    elif dataset == "ozone":
        """
        Variables : 8
        Timepoints: 203
        """
        filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/ozone.csv'
        df = pd.read_csv(filepath, nrows = sum(Ns))
        df = df.iloc[:, [3, 5, 6, 7, 8, 10, 11, 12]]
        X = df.to_numpy()
        X = X.T 
        
        
    X_train = X[..., :Ns[0]]
    X_val = X[..., Ns[0]:(Ns[1] + Ns[0])]
    X_test = X[..., -Ns[2]:]
    return X_train, X_val, X_test


def book_data(sample_size):
    """
    Creates a sample based on the coefficients of the book tsa4
    
    Input:
      sample_size: The number of observations to generate
      
    Return:
      X: The data as a numpy array
      A1, A2: The matrix coefficients as numpy arrays
    """
    np.random.seed(0)
    A1 = np.array([[.3, -.2, .04],
                   [-.11, .26, -.05],
                   [.08, -.39, .39]])

    A2 = np.array([[.28, -.08, .07],
                   [-.04, .36, -.1],
                   [-.33, .05, .38]])

    total = sample_size + 100
    X_total = np.zeros((3, total))
    
    e = np.random.normal(0, 1, (3, total))
    
    # X_total[..., 0:2] = np.array([[73.23, 67.59, 67.46], [73.23, 67.59, 67.46]]).T + e[..., 0:2]
    # for i in range(2, total):
    #     X_total[..., i] = np.array([73.23, 67.59, 67.46]) + \
    #                       np.dot(A1, X_total[..., i-1]) + \
    #                       np.dot(A2, X_total[..., i-2]) + e[..., i]
    
    X_total[..., 0:2] = e[..., 0:2]
    for i in range(2, total):
        X_total[..., i] = np.dot(A1, X_total[..., i-1]) + \
                          np.dot(A2, X_total[..., i-2]) + e[..., i]
     
        
    return X_total[..., -sample_size:]#, A1, A2


def book_data_1(sample_size):
    """
    Creates a sample based on the coefficients of the book tsa4
    
    Input:
      sample_size: The number of observations to generate
      
    Return:
      X: The data as a numpy array
      A1, A2: The matrix coefficients as numpy arrays
    """
    np.random.seed(0)
    total = sample_size + 20
    
    a = np.array([73.23, 67.59, 67.46])
    
    A = np.array([[ .46, -.36,  .10],
                  [-.24,  .49, -.13],
                  [-.12, -.48,  .58]])

    X_total = np.zeros((3, total))
    e = np.random.normal(0, 1, (3, total))
    X_total[..., 0] = a + e[..., 0]
    
    for i in range(1, total):
        X_total[..., i] = a + np.dot(A, X_total[..., i-1]) + e[..., i]
        
    return X_total[..., -sample_size:]


def compute_rmse(y_pred, y_true):
    """
    The function that computes and returns the rmse
    
    Input:
        y_pred: predicted values
        y_true: true values
    
    Returns:
        The rmse value
    """
    rmse = np.sqrt( np.linalg.norm(y_pred - y_true)**2 / np.size(y_true) )
    return rmse


def compute_nrmse(y_pred, y_true):
    """
    The function that computes and returns the nrmse
    
    Input:
        y_pred: predicted values
        y_true: true values
    
    Returns:
        The nrmse value
    """
    # t1 = np.linalg.norm(y_pred - y_true)**2 / np.size(y_true)
    t1 = compute_rmse(y_pred, y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    nrmse = t1 / t2
    return nrmse


def get_ranks(tensor):
    """
    The function that calculates and returns the ranks of each mode-d unfolding of a tensor.
    
    Input:
        tensor: The tensor which will be unfolded
    
    Returns:
        ranks: As a numpy array
    """
    ranks = []
    for i in range(len(tensor.shape) - 1):
        temp = tl.unfold(tensor, i)
        ranks.append( np.linalg.matrix_rank(temp) )
    return np.array(ranks)


def difference(data, order):
    """
    Calculates the d-th order differencing of an array
    
    Input:
        data: A numpy array
        order: The order of differencing
        
    Returns:
        data: The transformed array
        inv: A list containing the elements required for the inverse differencing process
    """
    inv = []
    for _ in range(order):
        #print("Difference", data[...,0].shape)
        inv.append(data[..., 0])
        data = np.diff(data)
    return data, inv


def inv_difference(data, inv, order):
    """
    Calculates the d-th order inverse differencing of an array
    
    Input:
        data: A numpy array
        inv: A list that contains the elements required for the inverse differencing process
        order: The order of differencing
        
    Returns:
        data: The transformed array
    """
    inv_values = copy.deepcopy(inv)
    for _ in range(order):
        t = inv_values.pop()
        t_shape = list(t.shape)
        t_shape.append(1)
        tmp = t.reshape(tuple(t_shape))
        if len(t_shape) == 2:
            data = np.cumsum(np.hstack([tmp, data]), axis=-1)
        elif len(t_shape) == 3:
            data = np.cumsum(np.dstack([tmp, data]), axis=-1)
    return data


def cointegration_test(df, alpha=0.05): 
    """
    Perform Johanson's Cointegration Test and Report Summary
    
    Input:
        df: The data as a pandas dataframe where each column is a time series
    """
    
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """
    Perform ADFuller to test for Stationarity of given series and print report
    
    Input:
        series: The data as a pandas dataframe where each column is a time series
    
    Returns:
        counter: The number of Stationary Time Series
    """    
    
    counter = 0
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    #print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    #print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    #print(f' Significance Level    = {signif}')
    #print(f' Test Statistic        = {output["test_statistic"]}')
    #print(f' No. Lags Chosen       = {output["n_lags"]}')

    #for key,val in r[4].items():
    #    print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
    #    print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
    #    print(f" => Series is Stationary.")
        counter = 1
    else:
    #    print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
    #    print(f" => Series is Non-Stationary.") 
        counter = 0
    return counter

