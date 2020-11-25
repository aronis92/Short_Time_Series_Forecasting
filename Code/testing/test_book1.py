import matplotlib.pyplot as plt
import numpy as np


def book_data_1(sample_size):
    """
    Creates a sample based on the coefficients of the book tsa4
    
    Input:
      sample_size: The number of observations to generate
      
    Return:
      X: The data as a numpy array
      A1, A2: The matrix coefficients as numpy arrays
    """
    np.random.seed(0)
    total = sample_size + 20
    
    a = np.array([73.23, 67.59, 67.46])
    
    A = np.array([[ .46, -.36,  .10],
                  [-.24,  .49, -.13],
                  [-.12, -.48,  .58]])

    X_total = np.zeros((3, total))
    e = np.random.normal(0, 1, (3, total))
    X_total[..., 0] = a + e[..., 0]
    
    for i in range(1, total):
        X_total[..., i] = a + np.dot(A, X_total[..., i-1]) + e[..., i]
        
    return X_total[..., -sample_size:]

data = book_data_1(sample_size = 50)

plt.figure(figsize = (12,5))
#plt.ylim(-0.001, 0.001)
plt.plot(data.T)
plt.show()