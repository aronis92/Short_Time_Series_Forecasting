import numpy as np
import random
import statsmodels.api as sm
import scipy as sp
import matplotlib.pyplot as plt

def plot_results(data, title):
    epoch = [i + 1 for i in range(len(data))]
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.plot(epoch[1:], data[1:])

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

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

def fit_ar(data, p):
    r = autocorrelation(data, p)
    R = sp.linalg.toeplitz(r[:p])
    r = r[1:]
    A = sp.linalg.pinv(R).dot(r)
    return A

def fit_ar_ma(data, p, q):
    N = data.shape[-1]
    A = fit_ar(data, p)
    B = [0.]
    if q > 0:
        Res = []
        for i in range(p, N):
            res = data[..., i] - np.sum([ a * data[..., i-j] for a, j in zip(A, range(1, p + 1))], axis=0)
            Res.append(res)
        Res = np.array(Res)
        B = fit_ar(Res, q)
    return A, B

'''Define Parameters'''
np.random.seed(0)
random.seed(0)
p = 2
q = 0
d = 0 
dim = 2
n_samples = 2
mu = 0
sigma = 1

'''Create Dataset'''
X = np.random.random_sample((dim, p))
#X = np.array([2, 3]).reshape((dim, p))
E = np.random.normal(mu, sigma, (dim, n_samples + q))
#E = np.array([0.1, 0.2, 0.3]).reshape((dim, n_samples + q))
#A = np.array([0.4563476, 0.5591154]).reshape((p, 1))
A = np.array([0.7, 0.3])
#A = np.random.normal(mu, sigma, (p, 1))
#A = np.array([1, 2]).reshape((p, 1))
A = np.flip(A)
if q>0:
    #B = np.array([-1.111807]).reshape((q, 1))
    B = np.random.normal(mu, sigma, (q, 1))
    #B = np.array([-0.5, 0.5]).reshape((q, 1))
    B = np.flip(B)       
for t in range(n_samples):
    #x_t = np.dot(X[:, t:(t + p)], A) - np.dot(E[:, t:(t+q)], B) + E[:, t + q].reshape(dim, 1)
    x_t = np.dot(X[:, t:(t + p)], A) + E[:, t + q].reshape(dim, 1)
    X = np.hstack((X, x_t))
if d != 0 :
    X_new = X[:, p + d:] + X[:, p:-d]
    X = X_new

r = autocorrelation(X, p)

A_pred = fit_ar(X, p)
#A_pred, B_pred = fit_ar_ma(X, p, q)

#arma_mod20 = sm.tsa.ARMA(X, (p, q)).fit(disp=False)
plot_results(X.T, "test")