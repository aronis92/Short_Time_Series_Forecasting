import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)
sample_size = 50


A1 = np.array([[.3, -.2, .04],
               [-.11, .26, -.05],
               [.08, -.39, .39]])

A2 = np.array([[.28, -.08, .07],
               [-.04, .36, -.1],
               [-.33, .05, .38]])

total = 2*sample_size

X_total = np.zeros((3, total))
X_total[..., 0:2] = np.random.rand(3,2)

for i in range(2, total):
    X_total[..., i] = np.dot(-1.5*A1, X_total[..., i-1]) + np.dot(-1.5*A2, X_total[..., i-2]) #+ np.random.rand(3,)
    


plt.figure(figsize = (12,5))
#plt.ylim(-0.001, 0.001)
plt.plot(X_total.T)
plt.show()

print(np.mean(X_total, axis=1))