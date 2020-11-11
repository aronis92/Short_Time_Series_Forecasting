import numpy as np
from random import seed
from math import log
import tensorly as tl
from statsmodels.tsa.arima_process import arma_generate_sample



def get_matrix_coeff_data(sample_size, n_rows, n_columns):
    """
    Creates a sample based on the coefficients of the book tsa4
    Input:
      sample_size: The number of observations to generate   
      n_rows: number of rows of the data matrix
      n_columns: number of columns of the data matrix
    Return:
      X: The data as a numpy array
      A1, A2: The matrix coefficients as numpy arrays
    """
    np.random.seed(42)
    seed(42)
    total = 2000

    X_total = np.zeros((n_rows*n_columns, total))
    X_total[..., 0:2] = log(np.random.rand(n_rows*n_columns, 2))
    #max_v = 2.5
    #min_v = 1.5
    A1 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    #A1 = A1/((min_v + random()*(max_v - min_v))*la.norm(A1, 'fro'))
    A2 = np.random.rand(n_rows*n_columns, n_rows*n_columns)
    #A2 = A2/((min_v + random()*(max_v - min_v))*la.norm(A2, 'fro'))

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






# TO DELETE
def train_predict(X_hat, Us, S_pinv, par, mod):
    p = par['p']
    # Estimate the core tensor with the given Us.
    G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
    
    A = fit_model(G, par['p'], mod) 
         
    # Predict the core
    G_pred = 0
    if mod == "AR":
        for i in range(p):
            G_pred += A[i] * G[..., -(i + 1)]
        X_pred = tl.tenalg.multi_mode_dot(G_pred, Us)
    
    elif mod == "myVAR":
        G_pred = A[0]
        for i in range(p):
            tmp1 = G[..., -(i + 1)].flatten()
            G_pred += np.dot(A[i + 1], tmp1)
        G_pred = G_pred.reshape(G.shape[0], G.shape[1])
        
    elif mod == "VAR":
        for i in range(p):
            tmp1 = G[..., -(i + 1)].flatten()
            tmp1 = tmp1.reshape(tmp1.shape[0], 1)
            tmp2 = np.dot(A[i], tmp1)
            tmp3 = tmp2.reshape(G.shape[0], G.shape[1])
            G_pred += tmp3
    
    X_pred = tl.tenalg.multi_mode_dot(G_pred, Us)

    dim_list = list(X_pred.shape)
    dim_list.append(1)
    X_hat = np.append(X_hat, X_pred.reshape( tuple(dim_list) ), axis = -1)

    pred_mat = tl.tenalg.mode_dot( tl.unfold(X_hat[..., 1:], 0), S_pinv, -1)
    pred_value = pred_mat[:, -1]
    return pred_value, A


def predict(mod, Us, A, par, X, X_test):
    p = par['p']
    X_hat, S_pinv = MDT(X, par['r'])
    # Estimate the core tensor with the given Us.
    G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
    # Predict the core
    G_pred = 0
    if mod == "AR":
        for i in range(p):
            G_pred += A[i] * G[..., -(i + 1)]
            
    elif mod == "VAR":
        for i in range(p):
            tmp1 = G[..., -(i + 1)].flatten()
            tmp1 = tmp1.reshape(tmp1.shape[0], 1)
            tmp2 = np.dot(A[i], tmp1)
            tmp3 = tmp2.reshape(G.shape[0], G.shape[1])
            G_pred += tmp3
            
    elif mod == "myVAR":
        G_pred = A[0]
        print("G_pred shape 1: ", G_pred.shape)
        for i in range(p):
            tmp1 = G[..., -(i + 1)].flatten()
            #tmp1 = tmp1.reshape(tmp1.shape[0], 1)
            print("tmp 1 shape: ", tmp1.shape)
            G_pred += np.dot(A[i + 1], tmp1)
            #print("tmp 2 shape: ", tmp2.shape)
            G_pred = G_pred.reshape(G.shape[0], G.shape[1])
    
    X_pred = tl.tenalg.multi_mode_dot(G_pred, Us)
            
    dim_list = list(X_pred.shape)
    dim_list.append(1)
    X_hat = np.append(X_hat, X_pred.reshape( tuple(dim_list) ), axis = -1)
    # Reverse differencing should happen here
    pred_mat = tl.tenalg.mode_dot( tl.unfold(X_hat[..., 1:], 0), S_pinv, -1)
    pred_value = pred_mat[:, -1]
    pred_value = pred_value.reshape(pred_value.shape[0], 1)
    
    rmse = compute_rmse(pred_value, X_test)
    nrmse = compute_nrmse(pred_value, X_test)
    return rmse, nrmse








