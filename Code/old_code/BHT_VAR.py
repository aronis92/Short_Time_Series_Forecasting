from functions.BHT_ARIMA_functions import initialize_Us, update_Um, compute_convergence
from functions.utils import create_synthetic_data, compute_nrmse, compute_rmse
from functions.ARIMA_functions import fit_ar
from functions.MDT_functions import MDT
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
import copy

def update_cores(m, p, A, Us, Xd, Gd, lam):
    temp_Gd = []
    outer = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= m ])
    
    for t in range(p, Xd.shape[-1]):
        v1 = Us[m].T
        v2 = tl.unfold( Xd[..., t], m)
        v3 = outer.T
        summary = lam * np.linalg.multi_dot( [v1, v2, v3] )
        
        if isinstance(A[0], np.ndarray):
            for i in range(p):
                summary += np.dot(A[i], tl.unfold(Gd[..., t - i - 1], m))
        else:
            for i in range(p):
                summary += A[i] * tl.unfold(Gd[..., t - i - 1], m)
        
        if summary.shape != Gd[..., 0].shape:
            summary = summary.T
        
        temp_Gd.append(summary/(1 + lam))
    
    if m == 1:
        temp_Gd = [tmp.T for tmp in temp_Gd]

    temp_Gd = np.transpose(np.array(temp_Gd), (1,2,0))
    n_updatable = temp_Gd.shape[-1]
    n_all = Gd.shape[-1]
    Gd[..., (n_all - n_updatable):] = temp_Gd
    return Gd

def predict(X_hat, Us, S_pinv, par, model):
    p = par['p']
    # Forecast the next core tensor.
    G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
    if model == fit_ar:
        A = model(G, p)
    elif model == VAR:
        mod = model(G.T)
        res = mod.fit(p)
        A = res.coefs
        A_list = []
        for i in range(A.shape[0]):
            A_list.append(A[i, ...])
        A = A_list
    # Predict the core
    G_pred = 0
    if isinstance(A[0], np.ndarray):
        for i in range(p):
            G_pred += np.dot(A[i], G[..., -(i + 1)])
    else:
        for i in range(p):
            G_pred += A[i] * G[..., -(i + 1)]       

    Xd_pred = tl.tenalg.multi_mode_dot(G_pred, Us)

    dim_list = list(Xd_pred.shape)
    dim_list.append(1)
    X_hat = np.append(X_hat, Xd_pred.reshape( tuple(dim_list) ), axis = -1)
    # Reverse differencing should happen here
    pred_mat = tl.tenalg.mode_dot( tl.unfold(X_hat[..., 1:], 0), S_pinv, -1)
    return pred_mat[:, -1]

def train(data, par, model, tensorization):
    Rs = np.array(par['ranks'])
    conv = 10
    epoch = 0
    convergences = list([])
    metrics = list([])
    X_train = data[..., :-1]
    X_test = data[..., -1]
    X_hat, S_pinv = tensorization(X_train, par['r'])
    # Us Initialization
    Us = initialize_Us(X_hat, Rs)
    # Train the AR model
    while epoch < par['max_epoch'] and conv > par['threshold']:
        old_Us = copy.deepcopy(Us)
        # Step 9 - Calculate the Core Tensors
        Gd = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
        # Calculate par of AR model
        
        if model == fit_ar:
            A = model(Gd, par['p'])
        elif model == VAR:
            mod = model(data.T) # [..., :-1]
            results = mod.fit(par['p'])
            A = results.coefs
            A_list = []
            for i in range(A.shape[0]):
                A_list.append(A[i, ...])
            A = A_list
        

        for m in range(len(Us)):
            # Step 12 - Update cores over m-unfolding
            Gd = update_cores(m, par['p'], A, Us, X_hat, Gd, par['lam'])
            # Step 13 - Update U[m]
            Us[m] = update_Um(m, par['p'], X_hat, Gd, Us)
        
        conv = compute_convergence(Us, old_Us)
        convergences.append(conv)
        epoch += 1
        
        # Update
        prediction = predict(X_hat, Us, S_pinv, par, model)

        rmse = compute_rmse(prediction, X_test)
        nrmse = compute_nrmse(prediction, X_test)
        metrics.append([rmse, nrmse])
        
    return Us, np.array(convergences), np.array(metrics)


np.random.seed(0)
X = create_synthetic_data(p = 2, q = 0, d = 0, dim = 5, n_samples = 40, mu = 0, sigma = 1)

parameters = {'r': 3,
              'p': 2,
              'ranks': [5,5],
              'lam': 1,
              'max_epoch': 10,
              'threshold': 0.001}

Us, convergences, changes = train(data = X,
                                  par = parameters,
                                  model = VAR,
                                  tensorization = MDT)










