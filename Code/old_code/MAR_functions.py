from functions.MDT_functions import MDT
from functions.utils import create_synthetic_data
import scipy as sp
import numpy as np

def get_corr_matrix(X, p):
    R = []
    m = np.mean(X, axis=-1)
    for i in range(p + 1):
        r = 0
        # count = 0
        for k in range(i, X.shape[-1] - i - 1):
            tmp = np.dot((X[..., k] - m).reshape((X.shape[0]*X.shape[1], 1)),
                         (X[..., k-i] - m).reshape((X.shape[0]*X.shape[1], 1)).T )
            r += tmp
        R.append(r/((X.shape[-1]))) #count) Isws allagh o paronomasths
    return R

def get_toeplitz_R(R, p):
    d = R[0].shape[0]
    R_hat = np.zeros((p*d, p*d))
    times = int( R_hat.shape[0]/d )
    for i in range(times):
        start = i*d
        end = start + d
        R_hat[start:end, start:end] = R[0]
        for j in range(times - i - 1):
            start2 = start + (j + 1)*d
            end2 = end + (j + 1)*d
            R_hat[start:end, start2:end2] = R[j + 1]
            R_hat[start2:end2, start:end] = R[j + 1].T
    return R_hat

def get_r(R, p):
    d = R[0].shape[0]
    r = np.zeros((d, p*d))
    for i in range(0, p*d, d):
        r[:, i:(i+d)] = R[1 + int(i/d)]
    return r

def fit_mar(X, p):
    R = get_corr_matrix(X, p)
    R_hat = get_toeplitz_R(R, p)
    #R_hat = sp.linalg.toeplitz(R[:p])
    # r = np.array(R[1:])
    r = get_r(R, p)
    A = np.dot(-r, sp.linalg.pinv(R_hat) )
    As = []
    for i in range(0, A.shape[-1], A.shape[0]):
        As.append( A[:, i:(i+R[0].shape[0])] )
    return As


# X = create_synthetic_data(p=2, dim=3, n_samples=100)
# X = X[:3, ...]
# X, _ = MDT(X, r=2)
# X_test = X[..., -1]
# X = X[..., :-1]

# p = 2
# A = fit_mar(X, p)

# X_pred = 0
# for i in range(p):
#     #X_pred -= A[i] * X[..., -(i + 1)]
#     X_pred -= np.dot(A[i], X[..., -(i + 1)].reshape((X.shape[0]*X.shape[1], 1)) )

# X_pred = X_pred.reshape((X.shape[0], X.shape[1]))









