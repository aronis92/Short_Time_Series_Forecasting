import numpy as np
import tensorly as tl

coeffs = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]],
                   [[9, 10], [11, 12]],
                   [[13, 14], [15, 16]]])

data = -np.array([[1, 2], [3, 4]])

coeffs = np.transpose(coeffs, (1,2,0))

coeffs_unfolded = tl.unfold(coeffs, mode=-1)

data_unfolded = data.reshape((data.shape[0]*data.shape[1], 1))

res = np.dot(coeffs_unfolded, data_unfolded)

coeffs_refolded = tl.fold(coeffs_unfolded, mode=-1, shape=coeffs.shape)

data_refolded = data.reshape((data.shape[0], data.shape[0]))