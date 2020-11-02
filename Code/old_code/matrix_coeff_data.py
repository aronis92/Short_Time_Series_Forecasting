import numpy as np
from random import random, seed
from numpy import linalg as la
from numpy import log
from functions.AR_functions import autocorrelation
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def plot_results(data, title, ytitle):
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


def matrix_coeff_data(order, sample_size, n_rows, n_columns):
    np.random.seed(69)
    total = 2000
    
    X_total = np.zeros((n_rows, n_columns, total))
    X_total[..., 0] = np.random.rand(n_rows, n_columns)
    
    if order == 1:
        A = np.array([[.46, -.36, .10],
                      [-.24, .49, -.13],
                      [-.12, -.48, .58]])
        #print(la.norm(A, 'fro'))
        for i in range(1, total):
            X_total[..., i] = np.dot(A, X_total[..., i-1]) + np.random.rand(n_rows, n_columns)
        
    elif order == 2:
        A_1 = np.array([[.3, -.2, .04],
                        [-.11, .26, -.05],
                        [.08, -.39, .39]])
        #print(la.norm(A_1, 'fro'))
        A_2 = np.array([[.28, -.08, .07],
                        [-.04, .36, -.1],
                        [-.33, .05, .38]])
        #print(la.norm(A_2, 'fro'))
        
        X_total[..., 1] = np.random.rand(n_rows, n_columns)
        for i in range(2, total):
            X_total[..., i] = np.dot(A_1, X_total[..., i-1]) + np.dot(A_2, X_total[..., i-2]) + np.random.rand(n_rows, n_columns)
    
    X = X_total[..., (total-sample_size):]
    X_vectorized = np.reshape(X, (n_rows*n_columns, X.shape[-1]))
    
    #plt.figure(figsize = (12,5))
    X_vectorized = X_vectorized[:5, :]
    #plt.plot(X_vectorized.T)
    return X


# np.random.seed(0)
# A = np.random.rand(a_size, a_size)
# A_norm = la.norm(A, 'fro')
# A = A/A_norm
# A_norm = la.norm(A, 'fro')
#X = matrix_coeff_data(order=2, sample_size=100, n_rows=3, n_columns=3)



def get_matrix_coeff_data(sample_size, n_rows, n_columns):
    np.random.seed(42)
    seed(42)
    total = 2000

    X_total = np.zeros((n_rows*n_columns, total))
    X_total[..., 0:2] = log(np.random.rand(n_rows*n_columns, 2))
    max_v = 3
    min_v = 2
    A1 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    A1 = A1/((min_v + random()*(max_v - min_v))*la.norm(A1, 'fro'))
    A2 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    A2 = A2/((min_v + random()*(max_v - min_v))*la.norm(A2, 'fro'))
    

    for i in range(2, total):
        X_total[..., i] = np.dot(A1, X_total[..., i-1]) + np.dot(A2, X_total[..., i-2]) + np.random.rand(n_rows*n_columns)
    
    
    X = X_total[..., (total-sample_size):]
    return log(X)


n_rows = 5
n_columns = 5
sample_size = 100

n_rows = 1
n_columns = 1
sample_size = 100
X = get_matrix_coeff_data(sample_size, n_rows, n_columns)
X_vectorized = np.reshape(X, (n_rows*n_columns, X.shape[-1]))

plt.figure(figsize = (16,5))
#X_vectorized = X_vectorized[5, :]
plt.plot(X_vectorized.T)

# np.mean(X_vectorized, axis=1)
# l = list([])
# for i in range(X_vectorized.shape[0]):
#     result = adfuller(X_vectorized[i,:])
#     print('ADF Statistic: %f' % result[0])
#     print('p-value: %f' % result[1])
#     print('Critical Values:')
#     for key, value in result[4].items():
#         print('\t%s: %.3f' % (key, value))
#         if key == "10%":
#             v = value
#             if result[0]<v:
#                 l.append(True)
#             else:
#                 l.append(False)
# print(sum(l))





















