import numpy as np
import random
import statsmodels.api as sm
import scipy as sp
import matplotlib.pyplot as plt

def plot_results(data, title):
    epoch = [i + 1 for i in range(len(data))]
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.plot(epoch, data)

def autocorrelation(data, p):
    # Calculate the autocorrelation for steps 0 - p (0, p-1) is needed for R and (1, p) for r
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

'''Define Parameters'''
np.random.seed(0)
random.seed(0)
p = 2
dim = 2
n_samples = 100
mu = 0
sigma = 1

X = np.random.random_sample((dim, p))
E = np.random.normal(mu, sigma, (dim, n_samples))
A = np.array([1, -0.9]).reshape((p, 1))
A = np.flip(A)

for t in range(n_samples):
    x_t = np.dot(X[:, t:(t + p)], A) + E[:, t].reshape((dim, 1))
    X = np.hstack((X, x_t))
del x_t, mu, sigma, dim, t
    
#plot_results(X.T, 'test')

#A_pred = fit_ar(X, p)
r = autocorrelation(X, p)
R = sp.linalg.toeplitz(r[:p])

A_pred = np.dot(sp.linalg.pinv(R), r[1:])

A_pred = np.flip(A_pred)