import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from functions.MDT_functions import MDT


np.random.seed(0)

def book_data_1(sample_size):

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

def book_data_2(sample_size):

    np.random.seed(0)
    total = sample_size + 20
    
    a = np.array([73.23, 67.59, 67.46])
        
    A = np.array([[ .46, -.36,  .10],
                  [-.24,  .49, -.13],
                  [-.12, -.48,  .58]])

    X_total = np.zeros((3, total))
   
    e1 = np.random.normal(0, 1, (1, 70))
    e2 = np.random.normal(0, 1, (1, 70))
    e3 = np.random.normal(0, 1, (1, 70))
    e = np.vstack((e1,e2,e3))
    
    X_total[..., 0] = a + e[..., 0]
    
    for i in range(1, total):
        X_total[..., i] = a + np.dot(A, X_total[..., i-1]) + e[..., i]
        
    return X_total[..., -sample_size:]

#X = book_data_1(50)
#plt.plot(X.T)
#plt.show()

#X2 = book_data_2(50)
#plt.plot(X2.T)
#plt.show()

#U1, _, V1 = np.linalg.svd(X.T, full_matrices=False)
#proc1 = np.dot(U1, V1)


X = book_data_1(5)
X_hat, S_pinv = MDT(X, 2)



X_unfolded = tl.unfold(X_hat, 0).T






























