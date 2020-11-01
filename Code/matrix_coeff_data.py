import numpy as np
from numpy import linalg as la
from functions.AR_functions import autocorrelation
import matplotlib.pyplot as plt

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

        for i in range(1, total):
            X_total[..., i] = np.dot(A, X_total[..., i-1]) + np.random.rand(n_rows, n_columns)
        
    elif order == 2:
        A_1 = np.array([[.3, -.2, .04],
                        [-.11, .26, -.05],
                        [.08, -.39, .39]])
        A_2 = np.array([[.28, -.08, .07],
                        [-.04, .36, -.1],
                        [-.33, .05, .38]])
        
        X_total[..., 1] = np.random.rand(n_rows, n_columns)
        for i in range(2, total):
            X_total[..., i] = np.dot(A_1, X_total[..., i-1]) + np.dot(A_2, X_total[..., i-2]) + np.random.rand(n_rows, n_columns)
        
    
    X = X_total[..., (total-sample_size):]
    X_vectorized = np.reshape(X, (n_rows*n_columns, X.shape[-1]))
    
    plt.figure(figsize = (12,5))
    X_vectorized = X_vectorized[:5, :]
    plt.plot(X_vectorized.T)
    return X


# np.random.seed(0)
# A = np.random.rand(a_size, a_size)
# A_norm = la.norm(A, 'fro')
# A = A/A_norm
# A_norm = la.norm(A, 'fro')



#acfa = autocorrelation(A, 0)
#acfb = autocorrelation(B, 0)
#Phi = np.kron(A,B)
    

X = matrix_coeff_data(order=2, sample_size=100, n_rows=3, n_columns=3)





