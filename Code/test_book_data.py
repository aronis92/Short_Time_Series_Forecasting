import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions.utils import adfuller_test

def semi_definite(matrixSize):
    from scipy import random, linalg
    A = random.rand(matrixSize,matrixSize)
    B = np.dot(A,A.transpose())
    return B

np.random.seed(0)
sample_size = 200
n_cut = 100

a = np.array([73.23, 67.59, 67.46])

A1 = np.array([[.3, -.2, .04],
               [-.11, .26, -.05],
               [.08, -.39, .39]])

A2 = np.array([[.28, -.08, .07],
               [-.04, .36, -.1],
               [-.33, .05, .38]])

total = 2*sample_size

X_total = np.zeros((3, total))
mean = np.zeros(3)
cov = semi_definite(3)
#X_total[..., 0:2] = np.random.rand(3,2)
X_total[..., 0:2] = np.random.multivariate_normal(mean, cov, 2).T

for i in range(2, total):
    X_total[..., i] = a + np.dot(-A1, X_total[..., i-1]) + np.dot(-A2, X_total[..., i-2])# + np.random.rand(3,)
    

plt.figure(figsize = (12,5))
#plt.ylim(-0.001, 0.001)
plt.plot(X_total.T)
plt.show()

plt.figure(figsize = (12,5))
#plt.ylim(-0.001, 0.001)
plt.plot(X_total[0, n_cut:].T)
plt.show()

plt.figure(figsize = (12,5))
#plt.ylim(-0.001, 0.001)
plt.plot(X_total[1, n_cut:].T)
plt.show()

plt.figure(figsize = (12,5))
#plt.ylim(-0.001, 0.001)
plt.plot(X_total[2, n_cut:].T)
plt.show()

df = pd.DataFrame(X_total[:, n_cut:].T)
counter = 0
indices = []
index = 0
for name, column in df.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    #print('\n')
print(counter, "/", X_total.shape[0], "Series are Stationary")


data = list()
for i in range(100):
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
data = np.array(data).T

plt.figure(figsize = (12,5))
#plt.ylim(-0.001, 0.001)
plt.plot(data.T)
plt.show()







