####################################################
##                                                ##
##  This file contains various utility functions  ##
##                                                ##
####################################################

from functions.utils import compute_rmse, compute_nrmse, get_ranks
#from functions.MAR_functions import fit_mar
from functions.AR_functions import fit_ar
from functions.MDT_functions import MDT
from statsmodels.tsa.api import VAR
import statsmodels
import tensorly as tl
import numpy as np
import copy


''' Initialization Functions '''
def initialize_Us(tensor, ranks):
    Us = [] # Empty list that will contain the Us Matrices.
    # Exclude the Time dimension since it is the only dimension not decomposed.
    for i in range(len(tensor.shape) - 1):
        # Generate a random array with shape (tensor_size_on_dim_i, rank_on_dim_i)
        Us.append( np.random.rand(tensor.shape[i], ranks[i]) ) 
    return Us

''' Update Functions '''
def update_cores(m, p, A, Us, X, Gd, lam, mod):
    temp_Gd = []
    outer = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= m ])
    
    if mod == "AR":
        for t in range(p, X.shape[-1]):
            v1 = Us[m].T
            v2 = tl.unfold( X[..., t], m)
            v3 = outer.T
            summary = lam * np.linalg.multi_dot( [v1, v2, v3] )
            
            for i in range(p):
                summary += A[i] * tl.unfold(Gd[..., t - i - 1], m)
            
            if summary.shape != Gd[..., 0].shape:
                summary = summary.T
            
            temp_Gd.append(summary/(1 + lam))
    elif mod == "VAR":
        for t in range(p, X.shape[-1]):
            v1 = Us[m].T
            v2 = tl.unfold( X[..., t], m)
            v3 = outer.T
            summary = lam * np.linalg.multi_dot( [v1, v2, v3] )
            
            if summary.shape != Gd[..., 0].shape:
                summary = summary.T
            
            for i in range(p):
                tmp1 = Gd[..., t - i - 1].flatten()
                tmp1 = tmp1.reshape((tmp1.shape[0], 1))
                tmp2 = np.dot(A[i], tmp1)
                tmp3 = tmp2.reshape((Gd.shape[0], Gd.shape[1]))
                # print(summary.shape, tmp3.shape)
                summary += tmp3
            
            temp_Gd.append(summary/(1 + lam))
    
    # if m == 1:
    #     temp_Gd = [tmp.T for tmp in temp_Gd]

    temp_Gd = np.transpose(np.array(temp_Gd), (1,2,0))
    n_updatable = temp_Gd.shape[-1]
    n_all = Gd.shape[-1]
    Gd[..., (n_all - n_updatable):] = temp_Gd
    return Gd

def update_Um(m, p, Xd, Gd, Us):
    Bs = []
    H = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= m ])
    for t in range(p, Xd.shape[-1]):
        unfold_X = tl.unfold(Xd[..., t], m)
        dot1 = np.dot(unfold_X, H.T)
        Bs.append(np.dot(dot1, tl.unfold(Gd[..., t], m).T ))
    b = np.sum(Bs, axis=0)
    np.random.seed(0)
    U1, _, V1 = np.linalg.svd(b, full_matrices=False)
    proc1 = np.dot(U1, V1)
    Us[m] = proc1
    return np.dot(U1, V1)

''' Metric Calculation Functions '''
def compute_convergence(new_Us, old_Us):
    Us_difference = [ new - old for new, old in zip(new_Us, old_Us)]
    a = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in Us_difference], axis=0)
    b = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_Us], axis=0)
    return a/b

def predict(X_hat, Us, S_pinv, par, mod):
    p = par['p']
    # Forecast the next core tensor.
    G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
    if mod == "AR":
        A = model(G, par['p'])
    elif mod == "VAR":
        # A = model(G, par['p'])
        
        Gd2 = G.reshape((G.shape[0]*G.shape[1], G.shape[2]))
        model = VAR(Gd2.T)
        results = model.fit(par['p'])
        A2 = results.coefs
        A = []
        for i in range(A2.shape[0]):
            A.append(A2[i, ...])
        
        
    # Predict the core
    G_pred = 0
    if mod == "AR":
        for i in range(p):
            G_pred += A[i] * G[..., -(i + 1)]
    elif mod == "VAR":
        for i in range(p):
            tmp1 = G[..., -(i + 1)].flatten()
            tmp1 = tmp1.reshape(tmp1.shape[0], 1)
            tmp2 = np.dot(A[i], tmp1)
            tmp3 = tmp2.reshape(G.shape[0], G.shape[1])
            G_pred += tmp3
    
    
    Xd_pred = tl.tenalg.multi_mode_dot(G_pred, Us)

    dim_list = list(Xd_pred.shape)
    dim_list.append(1)
    X_hat = np.append(X_hat, Xd_pred.reshape( tuple(dim_list) ), axis = -1)
    # Reverse differencing should happen here
    pred_mat = tl.tenalg.mode_dot( tl.unfold(X_hat[..., 1:], 0), S_pinv, -1)
    return pred_mat[:, -1], A

def train(data, par, model):
    # Rs = np.array(par['ranks'])
    conv = 10
    epoch = 0
    convergences = list([])
    metrics = list([])
    X_train = data[..., :-1]
    X_test = data[..., -1]
    X_hat, S_pinv = MDT(X_train, par['r'])
    Rs = get_ranks(X_hat)
    # Us Initialization
    Us = initialize_Us(X_hat, Rs)
    # Train the AR model
    while epoch < par['max_epoch'] and conv > par['threshold']:
        old_Us = copy.deepcopy(Us)
        # Step 9 - Calculate the Core Tensors
        Gd = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
        # Calculate par of AR model
        if model == "AR":
            A = fit_ar(Gd, par['p'])
        elif model == "VAR":
            # A = fit_mar(Gd, par['p'])
            
            Gd2 = Gd.reshape((Gd.shape[0]*Gd.shape[1], Gd.shape[2]))
            model = VAR(Gd2.T)
            results = model.fit(par['p'])
            A2 = results.coefs
            A = []
            for i in range(A2.shape[0]):
                A.append(A2[i, ...])
        
        for m in range(len(Us)):
            # Step 12 - Update cores over m-unfolding
            Gd = update_cores(m, par['p'], A, Us, X_hat, Gd, par['lam'], "VAR")
            # Step 13 - Update U[m]
            Us[m] = update_Um(m, par['p'], X_hat, Gd, Us)
        
        conv = compute_convergence(Us, old_Us)
        convergences.append(conv)
        epoch += 1
        
        prediction, A = predict(X_hat, Us, S_pinv, par, "VAR")

        rmse = compute_rmse(prediction, X_test)
        nrmse = compute_nrmse(prediction, X_test)
        metrics.append([rmse, nrmse])
        
    return Us, np.array(convergences), np.array(metrics), A, prediction















