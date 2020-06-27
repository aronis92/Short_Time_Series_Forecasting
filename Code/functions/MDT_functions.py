import numpy as np
import tensorly as tl

def get_duplication_matrix(r, T):
    S = np.vstack((np.eye(r), np.zeros((T-r, r))))
    for i in range(T- r):
        end = (i + 1)*r  
        temp1 = S[:-1, (end - r):end]
        temp2 = np.vstack((np.zeros((1, r)), temp1))
        S = np.hstack((S, temp2))
    return S.T

# def get_S_pinv(r, T):
#     S = get_duplication_matrix(r, T)
#     S_pinv = np.dot( np.linalg.inv( np.dot(S.T, S) ), S.T )
#     return S_pinv

def get_MDT_tensor_shape(shape, r, T):
    tensor_shape = [T - r + 1]
    for i in range(len(shape) - 1):
        tensor_shape.append(shape[i])
    tensor_shape.extend([r])
    return tuple(tensor_shape)

def MDT(X, r):
    T = X.shape[-1]
    tensor_shape = get_MDT_tensor_shape(X.shape, r, T)
    S = get_duplication_matrix(r, T)
    mode_T_product = tl.tenalg.mode_dot(X, S, 1)
    X = tl.fold(mode_T_product, 1, tensor_shape)
    return np.transpose(X, (1, 2, 0)), np.dot( np.linalg.inv( np.dot(S.T, S) ), S.T ) # np.linalg.pinv(S)