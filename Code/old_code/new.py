import tensorly as tl
import numpy as np
import scipy as sp
import copy
import random
import matplotlib.pyplot as plt
from tensorly.tenalg.proximal import procrustes
from sklearn.metrics import mean_squared_error
from math import sqrt
from MDT_functions import MDT
from ARIMA_functions import fit_ar, fit_ar_ma

def create_synthetic_data(p, q, d, dim, n_samples, mu, sigma):
    np.random.seed(0)
    random.seed(0)

    X = np.random.random_sample((dim, p))
    A = np.array([0.4563476, 0.5591154]).reshape((p, 1))
    # if q > 0:
    #     E = np.random.normal(mu, sigma, (n_samples + q, dim, q + 1))
    #     B = np.array([-1.111807, 0]).reshape((q, 1))
    for t in range( n_samples):
        x_t = np.dot(X[:, t:(t + p)], A) # - (q > 0) * (np.dot(E[t, :, :-1], B) + E[t, :, -1].reshape(dim, 1) )
        X = np.hstack((X, x_t))
    
    return X[:, -n_samples:], A

def initialize_Us(tensor, ranks):
    np.random.seed(0)
    Us = []
    for i in range(len(tensor.shape) - 1):
        Us.append( np.random.rand(tensor.shape[i], ranks[i]) )
    return Us

def update_cores(m, p, Us, Xd, Gd, lam):
    temp_Gd = []
    outer = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= m ])
    
    for t in range(p, Xd.shape[-1]):
        v1 = Us[m].T
        v2 = tl.unfold( Xd[..., t], m)
        v3 = outer.T
        summary = lam * np.linalg.multi_dot( [v1, v2, v3] )
        
        for i in range(p):
            summary += A[i] * tl.unfold(Gd[..., t - i - 1], m)
        
        if summary.shape != Gd[..., 0].shape:
            summary = summary.T
        
        temp_Gd.append(summary/(1 + lam))

    temp_Gd = np.transpose(np.array(temp_Gd), (1,2,0))
    n_updatable = temp_Gd.shape[-1]
    n_all = Gd.shape[-1]
    Gd[..., (n_all - n_updatable):] = temp_Gd
    return Gd

def update_Um(m, p, Xd, Gd):
    Bs = []
    H = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= m ])
    for t in range(p, Xd.shape[-1]):
        unfold_X = tl.unfold(Xd[..., t], m)
        dot1 = np.dot(unfold_X, H.T)
        #dot1 = np.dot(Xd[..., t], H.T)
        Bs.append(np.dot(dot1, tl.unfold(Gd[..., t], m))) # Gd or Gd.T
        #Bs.append(np.dot(dot1, Gd[..., t].T))
    b = np.sum(Bs, axis=0)
    #b = b.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
    U_, _, V_ = np.linalg.svd(b, full_matrices=False)
    return np.dot(U_, V_)

def predict(p, Gd, Us, X_hat, A):
    # Predict the core G for the T_hat + 1 timestep
    Gd_pred = 0
    for i in range(p):
        Gd_pred += A[i] * Gd[..., -(i + 1)]
        
    # Convert the core to the original X for the T_hat + 1 timestep
    Xd_pred = tl.tenalg.multi_mode_dot(Gd_pred, Us, modes = [i for i in range(len(Us))], transpose = False)
    
    return Xd_pred

def tensor_difference(d, tensors):
        """
        
        get d-order difference series
        
        Arg:
            d: int, order
            tensors: list of ndarray, tensor to be difference
        
        Return:
            begin_tensors: list, the first d elements, used for recovering original tensors
            d_tensors: ndarray, tensors after difference

        """
        d_tensors = tensors
        tup = list(tensors.shape)
        tup[-1] = d
        begin_tensors = np.zeros(tuple(tup))

        for i in range(d):
            begin_tensors[..., i] = d_tensors[..., 0]
            d_tensors = np.diff(d_tensors, axis = -1)
        
        return begin_tensors, d_tensors

def inverse_MDT(p, Xd_pred, X_hat, S_pinv):
    X_hat_new = np.empty( tuple( list(X_hat.shape) ) )
    X_hat_new[..., :-1] = X_hat[..., 1:]
    X_hat_new[..., -1] = Xd_pred + X_hat[..., -p]
    
    predictions = tl.tenalg.mode_dot( tl.unfold(X_hat_new, 0), S_pinv, -1)
    return predictions[..., -1]

def plot_results(data, title):
    epoch = [i + 1 for i in range(len(data))]
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.plot(epoch[1:], data[1:])

np.random.seed(0)
X = np.load('traffic_40.npy').T
X, A_real = create_synthetic_data(2, 0, 0, 5, 40, 0, 1)
# Set some variables
idx = {'r': 5, 'p': 2, 'd': 0, 'q': 0, 'lam': 1, 'max_epoch': 100, 'threshold': 0.001}

X_train = X[:,:-1]
X_test = X[:,-1]
Rs = np.array([5, 5])
conv = 10
epoch = 0
convergences = list([])
pred_diff = list([])

# Apply MDT on the training Data and obtain the S and S_pinv
X_hat, S, S_pinv = MDT(X_train, idx['r'])
Xd = X_hat
Us = initialize_Us(X_hat, Rs)

# Apply Differencing to make stationary
#Xd = np.diff(X_hat, n = idx['d'], axis = -1) if idx['d'] > 0 else X_hat
#begin_X, Xd = tensor_difference(idx['d'], X_hat)

# Es Initialization Here

while epoch < idx['max_epoch'] and conv > idx['threshold']:
    new_Us = [[],[]]
    denominator = 0
    numerator = 0
    # Step 9 - Calculate the Core Tensors
    Gd = tl.tenalg.multi_mode_dot(Xd, Us, modes = [i for i in range(len(Us))], transpose = True)
    
    if idx['q'] == 0:
        A = fit_ar(Gd, idx['p'])
    else: 
        A, B = fit_ar_ma(Gd, idx['p'], idx['q'])
    
    for m in range(len(Us)):
        # Step 12 - Update cores over m-unfolding
        Gd = update_cores(m, idx['p'], Us, Xd, Gd, idx['lam'])
        # Step 13 - Update U[m]
        new_Us[m] = update_Um(m, idx['p'], Xd, Gd)
        
        numerator += np.linalg.norm( new_Us[m] - Us[m], ord = 'fro')**2
        denominator += np.linalg.norm( new_Us[m], ord = 'fro')**2
    
    Us = copy.deepcopy(new_Us)
    conv = numerator/denominator
    convergences.append(conv)
    epoch += 1
    
    # Predict the next value of Xd
    X_pred = predict(idx['p'], Gd, Us, X_hat, A)
    pred = inverse_MDT(idx['p'], X_pred, X_hat, S_pinv)
    # Append to the data in order to restore the differencing
    # if idx['d'] > 0:
    #     tup = list(X_hat.shape)
    #     tup[-1] += 1
    #     X_hat_new = np.zeros(tuple(tup))
    #     X_hat_new[..., :-1] = X_hat
    #     X_hat_new[..., -1] = X_pred
        
    #     X_new = tensor_reverse_diff(idx['d'], begin_X, X_hat_new)
    error = pred - X_test
    rmse = np.linalg.norm(error) / sqrt( np.size(error) )
    pred_diff.append( rmse )
    
plot_results(pred_diff, "RMSE")
    
    
    
test = np.array([i for i in range(30)])
test = test.reshape((5,6))
U1, _, V1 = np.linalg.svd(test, full_matrices = False)
    
    
    
    
    
    
    