from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
import random

def create_synthetic_data_old(p, dim, n_samples, mu, sigma):
    X = np.random.random_sample((dim, p))
    A = np.array([-0.3563476, 0.6591154]).reshape((p, 1))
    for t in range(n_samples):
        x_t = A[0]*X[..., -1] + A[1]*X[..., -2] + np.random.random_sample((dim,))*3 #np.dot(X[:, t:(t + p)], A) 
        X = np.hstack((X, x_t.reshape(dim,1)))
    return X, A

# X, A = create_synthetic_data_old(p=2, dim=5, n_samples=30, mu=0, sigma=1)
# plt.figure(figsize=(12,5))
# plt.plot(X.T)

def my_data(p, dim, n_samples):
    A = np.array([0.1, 0.9]).reshape((p, 1))
    X = np.random.random_sample((dim, p))
    noise = np.random.random_sample((dim, n_samples))
    for t in range(n_samples):
        x_t = np.dot(X[:, t:(t + p)], A) + noise[:, t] # A[0]*X[..., -1] + A[1]*X[..., -2] 
        X = np.hstack((X, x_t))
    return X[..., -n_samples:]

# X = my_data(p=2, dim=5, n_samples=30)
# plt.figure(figsize=(12,5))
# plt.plot(X.T)

def create_synthetic_data(p, dim, n_samples):
    np.random.seed(0)
    arparams = np.array([.75, -.25])
    ar = np.r_[1, -arparams]
    ma = np.array([1])
    extra = 10
    X = np.zeros([dim, extra*n_samples])
    for i in range(dim):
        X[i, :] = arma_generate_sample(ar, ma, nsample = extra*n_samples)
    return X[:, -n_samples:]

# X = create_synthetic_data(p=2, dim=5, n_samples=30)
# plt.figure(figsize=(12,5))
# plt.plot(X.T)

def my_data2(p, dim, n_samples):
    X_final = np.zeros((dim, n_samples))
    A = np.array([-0.2, 0.8]).reshape((p, 1))
    for d in range(dim):
        X = np.random.random_sample((1, p)) + np.random.randint(5, size=(1, p)) + 2
        noise = np.random.random_sample((1, n_samples))
        for t in range(n_samples):
            x_t = np.dot(X[:, t:(t + p)], A) + noise[:, t]
            X = np.hstack((X, x_t))
        X_final[d, ...] = X[..., -n_samples:]
    return X_final

# X = my_data2(p=2, dim=5, n_samples=30)
# plt.figure(figsize=(12,5))
# plt.plot(X.T)

def compute_rmse(y_pred, y_true):
    return np.sqrt( np.linalg.norm(y_pred - y_true)**2 / np.size(y_true) )

def compute_nrmse(y_pred, y_true):
    # t1 = np.linalg.norm(y_pred - y_true)**2 / np.size(y_true)
    t1 = compute_rmse(y_pred, y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return t1 / t2

def get_ranks(tensor):
    ranks = []
    for i in range(len(tensor.shape) - 1):
        temp = tl.unfold(tensor, i)
        ranks.append( np.linalg.matrix_rank(temp) )
    return np.array(ranks)








