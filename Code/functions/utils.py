####################################################
##                                                ##
##  This file contains various utility functions  ##
##                                                ##
####################################################

from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
from random import random, seed
from numpy import linalg as la
from numpy import log
import tensorly as tl
import numpy as np


# Plots the results
# Input:
#   data: The data to plot
#   title: The title of the graph
def plot_results(data, title, ytitle):
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



# Creates a sample based on the coefficients of the book tsa4
# Input:
#   sample_size: The number of observations to generate
# Return:
#   X: The data as a numpy array
#   A1, A2: The matrix coefficients as numpy arrays
def book_data(sample_size):
    np.random.seed(69)
    A1 = np.array([[.3, -.2, .04],
                   [-.11, .26, -.05],
                   [.08, -.39, .39]])
    #print(la.norm(A_1, 'fro'))
    A2 = np.array([[.28, -.08, .07],
                   [-.04, .36, -.1],
                   [-.33, .05, .38]])
    #print(la.norm(A_2, 'fro'))
    total = sample_size + 2000
    
    X_total = np.zeros((3, total))
    X_total[..., 0:2] = np.random.rand(3,2)
    for i in range(2, total):
        X_total[..., i] = np.dot(-A1, X_total[..., i-1]) + np.dot(-A2, X_total[..., i-2]) + np.random.rand(3,)
        
    return X_total[..., (total-sample_size):], A1, A2



# Creates a sample based on the coefficients of the book tsa4
# Input:
#   sample_size: The number of observations to generate   
#   n_rows: number of rows of the data matrix
#   n_columns: number of columns of the data matrix
# Return:
#   X: The data as a numpy array
#   A1, A2: The matrix coefficients as numpy arrays
def get_matrix_coeff_data(sample_size, n_rows, n_columns):
    np.random.seed(42)
    seed(42)
    total = 2000

    X_total = np.zeros((n_rows*n_columns, total))
    X_total[..., 0:2] = log(np.random.rand(n_rows*n_columns, 2))
    max_v = 2.5
    min_v = 1.5
    A1 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    A1 = A1/((min_v + random()*(max_v - min_v))*la.norm(A1, 'fro'))
    A2 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    A2 = A2/((min_v + random()*(max_v - min_v))*la.norm(A2, 'fro'))

    for i in range(2, total):
        X_total[..., i] = np.dot(A1, X_total[..., i-1]) + np.dot(A2, X_total[..., i-2]) + np.random.rand(n_rows*n_columns)
    
    
    X = X_total[..., (total-sample_size):]
    return X, A1, A2



# The function that creates and returns the simulation data
# Input:
#   p: AR model order
#   dim: dimensionality of data
#   n_samples: number of samples to create
# Returns:
#   data: A numpy matrix
def create_synthetic_data(p, dim, n_samples):
    np.random.seed(0)
    arparams = np.array([.75, -.25])
    ar = np.r_[1, -arparams]
    ma = np.array([1])
    extra = 10
    X = np.zeros([dim, extra*n_samples])
    for i in range(dim):
        X[i, :] = arma_generate_sample(ar, ma, nsample = extra*n_samples)
    data = X[:, -n_samples:]
    return data



# The function that creates and returns the simulation data
# Input:
#   p: AR model order
#   dim: dimensionality of data
#   n_samples: number of samples to create
# Returns:
#   data: A numpy matrix
def create_synthetic_data2(p, dim, n_samples):
    X = np.random.random_sample((dim, p))
    A = np.array([0.6, 0.3]).reshape((p, 1))
    for t in range(n_samples):
        x_t = A[0]*X[..., -1] + A[1]*X[..., -2] + np.random.random_sample((dim,))/2
        X = np.hstack((X, x_t.reshape(dim,1)))
    return X[:, -n_samples:]



# The function that computes and returns the rmse
# Input:
#   y_pred: predicted values
#   y_true: true values
# Returns:
#   The rmse value 
def compute_rmse(y_pred, y_true):
    rmse = np.sqrt( np.linalg.norm(y_pred - y_true)**2 / np.size(y_true) )
    return rmse



# The function that computes and returns the rmse
# Input:
#   y_pred: predicted values
#   y_true: true values
# Returns:
#   The nrmse value
def compute_nrmse(y_pred, y_true):
    # t1 = np.linalg.norm(y_pred - y_true)**2 / np.size(y_true)
    t1 = compute_rmse(y_pred, y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    nrmse = t1 / t2
    return nrmse



# The function that calculates and returns the ranks of each mode-d unfolding of a tensor.
# Input:
#   tensor: The tensor which will be unfolded
# Returns:
#   ranks: As a numpy array
def get_ranks(tensor):
    ranks = []
    for i in range(len(tensor.shape) - 1):
        temp = tl.unfold(tensor, i)
        ranks.append( np.linalg.matrix_rank(temp) )
    return np.array(ranks)







