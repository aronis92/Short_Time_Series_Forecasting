import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions.utils import adfuller_test

np.random.seed(0)
sample_size = 100

A1 = np.array([[.3, -.2, .04],
               [-.11, .26, -.05],
               [.08, -.39, .39]])
A2 = np.array([[.28, -.08, .07],
               [-.04, .36, -.1],
               [-.33, .05, .38]])
#a = np.dot(np.eye(3) - A1 - A2 , 0.5*np.ones(3))


total = sample_size + 50
X_total = np.zeros((3, total))


e = np.random.normal(0, 1, (3,total - 2))

X_total[..., 0:2] = np.random.rand(3,2)#np.random.normal(0, 1, (3,2))
for i in range(2, total):
    X_total[..., i] = np.dot(-A1, X_total[..., i-1]) + np.dot(-A2, X_total[..., i-2]) + e[..., i - 2]
X_total = X_total[..., -sample_size:]
    
plt.figure(figsize = (12,5))
plt.plot(X_total.T)
plt.show()

# plt.figure(figsize = (12,5))
# plt.plot(X_total[0, n_cut:].T)
# plt.show()

# plt.figure(figsize = (12,5))
# plt.plot(X_total[1, n_cut:].T)
# plt.show()

# plt.figure(figsize = (12,5))
# plt.plot(X_total[2, n_cut:].T)
# plt.show()

# n_cut = 50
# df = pd.DataFrame(X_total[:, n_cut:].T)
# counter = 0
# indices = []
# index = 0
# for name, column in df.iteritems():
#     c = adfuller_test(column, name=column.name)
#     if c==0:
#         indices.append(index)
#     index += 1
#     counter += c
#     #print('\n')
# print(counter, "/", X_total.shape[0], "Series are Stationary")

# print(np.mean(X_total, axis=1))


# e2 = np.zeros(e.shape)
# for i in range(e.shape[1]):
#     e2[:, i] = X_total[:, i + 2] + np.dot(A1, X_total[..., i+1]) + np.dot(A2, X_total[..., i])

# e3 = e-e2

from functions.utils import book_data

X, _, _ = book_data(sample_size = 100)
plt.figure(figsize = (12,5))
plt.plot(X.T)
plt.show()

