import numpy as np
import random
# import statsmodels.api as sm
import matplotlib.pyplot as plt

def create_synthetic_data(p, q, d, dim, n_samples, mu, sigma):
    np.random.seed(0)
    random.seed(0)

    X = np.random.random_sample((dim, p))
    E = np.random.normal(mu, sigma, (n_samples + q, dim, q + 1))
    A = np.array([0.4563476, 0.5591154]).reshape((p, 1))
    B = np.array([-1.111807, 0]).reshape((q, 1))

    for t in range( n_samples):
        x_t = np.dot(X[:, t:(t + p)], A) - np.dot(E[t, :, :-1], B) + E[t, :, -1].reshape(dim, 1)
        X = np.hstack((X, x_t))
    
    X_new = X[:, p + d:] + X[:, p:-d]
    return X[:, -n_samples:], X_new

X, X_new = create_synthetic_data(p=2, q=2, d=1, dim=1, n_samples=100, mu=0, sigma=1)

# y = sm.tsa.arma_generate_sample(A, B, 10, scale = 1)

def plot_results(data, title):
    epoch = [i + 1 for i in range(len(data))]
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.plot(epoch, list(data))
    
plot_results(X[:, 2:].reshape(X[:, 2:].shape[1],), 'Sample')
plot_results(X_new.reshape(X_new.shape[1],), 'Sample')

# es = [ [ np.random.random([36, 4]) for _ in range(3)] for t in range(5)]
# es0 = es[0]