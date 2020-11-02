#################################################################################
##                                                                             ##
##  This file contains the functions needed for the AR coefficient estimation  ##
##                                                                             ##
#################################################################################

import numpy as np
import scipy as sp


# The function that calculates and returns the autocorrelation list
# Input:
#   data: the data to calculate the autocorrelation from
#   p: lag value
# Returns:
#   autocorrelations in the form of a list r
def autocorrelation(data, p):
    T = data.shape[-1]
    r = []
    for l in range(p + 1):
        product = 0
        for t in range(T):
            tl = l - t if t < l else t - l
            product += np.sum(data[..., t] * data[..., tl])
        r.append(product)
    return r


# The function that calculates and returns the AR coefficients
# Input:
#   data: the data to calculate the autocorrelation from
#   p: lag value
# Returns:
#   AR coefficients as a list
def fit_ar(data, p):
    r = autocorrelation(data, p)
    R = sp.linalg.toeplitz(r[:p])
    r = r[1:]
    A = sp.linalg.pinv(R).dot(r)
    return A



def estimate_matrix_coefficients(data, p):
    k = data.shape[0]
    T = data.shape[1]
    Y = data[:, p:]
    Z = np.zeros((k*p+1, T-p))
    Z[0,:] = 1
    
    for i in range(1, p+1):
        Z[(k*(i-1)+1):(k*i+1), :] = data[:, (p-i):(-i)]
        
    B = np.linalg.multi_dot([Y, Z.T, np.linalg.inv(np.dot(Z, Z.T))])

    c = B[:, 0]
    A = [c]
    for i in range(p):
        A.append(B[:, (k*i+1):(k*i+1+k)])
    
    return A

# def fit_ar_ma(data, p, q):
#     N = data.shape[-1]
#     A = fit_ar(data, p)
#     B = [0.]
#     if q > 0:
#         Res = []
#         for i in range(p, N):
#             res = data[..., i] - np.sum([ a * data[..., i-j] for a, j in zip(A, range(1, p + 1))], axis=0)
#             Res.append(res)
#         Res = np.array(Res)
#         Res = np.transpose(Res, (1, 2, 0))
#         B = fit_ar(Res, q)
#     return A, B