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

def get_coeff_matrices(a_size, b_size):
    np.random.seed(1)
    A = np.random.rand(a_size, a_size)
    A_norm = la.norm(A, 'fro')
    A = A/A_norm
    #A_norm2 = la.norm(A, 'fro')
    B = np.random.rand(b_size, b_size)
    B_norm = la.norm(B, 'fro')
    B = B/(np.sqrt(2)*B_norm)
    #B_norm2 = la.norm(B, 'fro')
    return A, B
    
a_size = 1
b_size = 1
A, B = get_coeff_matrices(a_size, b_size)

#acfa = autocorrelation(A, 0)
#acfb = autocorrelation(B, 0)
#Phi = np.kron(A,B)
    
total = 300

X_total = np.zeros((a_size, b_size, total))
X_total[..., 0] = np.random.rand(a_size, b_size)

for i in range(1, total):
    X_total[..., i] = la.multi_dot([A, X_total[..., i-1], B.T]) + np.random.rand(a_size, b_size)

X_vectorized = np.reshape(X_total, (a_size*b_size, total))
X_vectorized = X_vectorized[..., 250:]

plt.figure(figsize = (12,5))
plt.plot(X_vectorized.T)