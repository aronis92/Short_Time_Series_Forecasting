###############################################################################
##                                                                           ##
##  This file contains the functions needed to apply MDT on a given dataset  ##
##                                                                           ##
###############################################################################

import numpy as np
import tensorly as tl


# The function that calculates and returns the duplication matrix S
# Input:
#   r: MDT Rank
#   T: Number of time points
# Returns:
#   duplication matrix as a numpy array
def get_duplication_matrix(r, T):
    S = np.vstack((np.eye(r), np.zeros((T-r, r))))
    for i in range(T- r):
        end = (i + 1)*r  
        temp1 = S[:-1, (end - r):end]
        temp2 = np.vstack((np.zeros((1, r)), temp1))
        S = np.hstack((S, temp2))
    return S.T


# The function that calculates and returns the shape of the tensor produced by MDT
# Input:
#   r: MDT Rank
#   T: Number of time points
#   shape: shape of the original dataset
# Returns:
#   tensor_shape as a tuple
def get_MDT_tensor_shape(shape, r, T):
    tensor_shape = [T - r + 1]
    for i in range(len(shape) - 1):
        tensor_shape.append(shape[i])
    tensor_shape.extend([r])
    return tuple(tensor_shape)


# The function that calculates and returns the MDT form of the data and the pseudoinverse of the duplication matrix S
# Input:
#   r: MDT Rank
#   X: original data
# Returns:
#   mdt_data as a numpy tensor
#   S_pinv as a numpy matrix
def MDT(X, r):
    T = X.shape[-1]
    tensor_shape = get_MDT_tensor_shape(X.shape, r, T)
    S = get_duplication_matrix(r, T)
    mode_T_product = tl.tenalg.mode_dot(X, S, 1)
    mdt_data = tl.fold(mode_T_product, 1, tensor_shape)
    mdt_data = np.transpose(mdt_data, (1, 2, 0))
    S_pinv = np.dot( np.linalg.inv( np.dot(S.T, S) ), S.T )
    return mdt_data, S_pinv # np.linalg.pinv(S)


