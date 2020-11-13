####################################################
##                                                ##
##  This file contains various utility functions  ##
##                                                ##
####################################################

from functions.utils import compute_rmse, compute_nrmse, get_ranks, difference, inv_difference
from functions.AR_functions import fit_ar, estimate_matrix_coefficients
from functions.MDT_functions import MDT
from statsmodels.tsa.api import VAR
import tensorly as tl
import numpy as np
import copy



def initialize_Us(tensor, ranks):
    """
    Ιnitialize and return the U matrices
    
    Input:
      - tensor: The data tensor
      - ranks: The rank of each unfolding of the tensor
    
    Returns:
      - Us: List containing each U matrix
    """
    Us = []
    # Exclude the Time dimension (-1) since it is the only dimension not decomposed.
    for i in range(len(tensor.shape) - 1):
        # Generate a random array with shape (tensor_size_on_dim_i, rank_on_dim_i)
        Us.append( np.random.rand(tensor.shape[i], ranks[i]) ) 
    return Us


##########################
##                      ##
##   Update Functions   ##
##                      ##
##########################

def update_cores(m, p, A, Us, X, G, lam, mod):
    """
    The function that updates and returns the core tensors
    
    Input:
      - m: The mode along which the update will take place
      - p: AR model order
      - A: Coeffients of the AR model
      - Us: List containing the U matrices
      - X: Data in their original form (before decomposition)
      - G: Previous core tensors
      - lam: Reguralization parameter
      - mod: Model to be used
    
    Returns:
      - G: New core tensors
    """
    temp_G = []
    outer = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= m ])
    
    if mod == "AR":
        for t in range(p, X.shape[-1]):
            v1 = Us[m].T
            v2 = tl.unfold( X[..., t], m)
            v3 = outer.T
            summary = lam * np.linalg.multi_dot( [v1, v2, v3] )
            
            for i in range(p):
                summary += A[i] * tl.unfold(G[..., t - i - 1], m)
            
            # if summary.shape != G[..., 0].shape:
            #     summary = summary.T
            summary = tl.fold(unfolded_tensor = summary, mode = m, shape = G[..., 0].shape)
            
            temp_G.append(summary/(1 + lam))
            
    elif mod == "VAR":
        for t in range(p, X.shape[-1]):
            v1 = Us[m].T
            v2 = tl.unfold( X[..., t], m)
            v3 = outer.T
            summary = lam * np.linalg.multi_dot( [v1, v2, v3] )
            
            
            # if summary.shape != G[..., 0].shape:
            #     summary = summary.T
            
            for i in range(p):
                tmp1 = G[..., t - i - 1].reshape((G.shape[0]*G.shape[1], 1))
                tmp1 = tmp1.reshape((tmp1.shape[0], 1))
                tmp2 = np.dot(A[i], tmp1)
                tmp3 = tmp2.reshape((G.shape[0], G.shape[1]))
                tmp3 = tl.unfold(tmp3, m)
                summary += tmp3
            
            summary = tl.fold(unfolded_tensor = summary, mode = m, shape = G[..., 0].shape)
            
            temp_G.append(summary/(1 + lam))
    
    # if m == 1:
    #     temp_Gd = [tmp.T for tmp in temp_Gd]

    temp_G = np.transpose(np.array(temp_G), (1,2,0))
    n_updatable = temp_G.shape[-1]
    n_all = G.shape[-1]
    G[..., (n_all - n_updatable):] = temp_G
    return G




def update_Um(m, p, X, G, Us):
    """
    Update and return the U matrix of mode m
    
    Input:
      - m: The mode along which the update will take place
      - p: AR model order
      - X: Data in their original form (before decomposition)
      - G: Core tensors
      - Us: List containing the U matrices
    
    Returns:
      - new_Um: New U matrix along mode-m
    """
    Bs = []
    H = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= m ])
    for t in range(p, X.shape[-1]):
        unfold_X = tl.unfold(X[..., t], m)
        dot1 = np.dot(unfold_X, H.T)
        Bs.append(np.dot(dot1, tl.unfold(G[..., t], m).T ))
    b = np.sum(Bs, axis=0)
    np.random.seed(0)
    U1, _, V1 = np.linalg.svd(b, full_matrices=False)
    proc1 = np.dot(U1, V1)
    Us[m] = proc1
    new_Um = np.dot(U1, V1)
    return new_Um




######################################
##                                  ##
##   Metric Calculation Functions   ##
##                                  ##
######################################

def compute_convergence(new_Us, old_Us):
    """
    Compute and return the convergence criterion value
    
    Input:
      - new_Us: The new U values
      - old_Us: The old U values
    
    Returns:
      - convergence_value: The value to be checked with the convergence criterion
    """
    Us_difference = [ new - old for new, old in zip(new_Us, old_Us)]
    a = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in Us_difference], axis=0)
    b = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_Us], axis=0)
    convergence_value = a/b
    return convergence_value




###########################################
##                                       ##
##   Training and Prediction functions   ##
##                                       ##
###########################################

def fit_model(data, p, mod):
    """
    The function that estimates the coefficients of the AR model.
    
    Input:
      - data: The loaded dataset
      - p: The order of the AR model
      - mod: A string "AR" or "VAR" that selects the model to be used
    
    Returns:
      - A: The coefficients (scalar or matrix)
    """
    if mod == "AR":
        A = fit_ar(data, p)
            
    elif mod == "myVAR":
        data_vectorized = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        A = estimate_matrix_coefficients(data_vectorized, p)
        A = A[1:]
        
    elif mod == "VAR":           
        data_vectorized = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        model = VAR(data_vectorized.T)
        results = model.fit(p)
        A2 = results.coefs
        A = []
        for i in range(A2.shape[0]):
            A.append(A2[i, ...])
            
    return A




def forecast(data, p, A, mod, n_forecast):
    """
    The function that implements and trains the whole algorithm.
    
    Input:
      - data: The loaded dataset
      - p: the order of the AR model
      - A: The coefficients of the AR model
      - mod: A string "AR" or "VAR" that selects the model to be used
    
    Returns:
      - prediction: The predicted values of the next step
    """
    predictions_list = list([])
    
    for j in range(n_forecast):
        prediction = 0
        
        if mod == "AR":
            for i in range(p):
                prediction += A[i] * data[..., -(i + 1)]
        
        elif mod == "myVAR":
            prediction = A[0]
            for i in range(p):
                tmp1 = data[..., -(i + 1)].flatten()
                prediction += np.dot(A[i + 1], tmp1)
            prediction = prediction.reshape(data.shape[0], data.shape[1])
            
        elif mod == "VAR":
            for i in range(p):
                #print("data shape: ", data[..., -(i + 1)].shape)
                tmp1 = data[..., -(i + 1)].flatten()
                #print("tmp1 shape: ", tmp1.shape)
                tmp1 = tmp1.reshape(tmp1.shape[0], 1)
                #print("tmp1 reshaped: ", tmp1.shape)
                tmp2 = np.dot(A[i], tmp1)
                tmp3 = tmp2.reshape(data.shape[0], data.shape[1])
                prediction += tmp3
        

        dim_list = list(prediction.shape)
        dim_list.append(1)
        data = np.append(data, prediction.reshape(tuple(dim_list)), axis=-1)  
        predictions_list.append(prediction)
    
    return predictions_list




def BHTAR(data_train, data_val, par, mod):
    """
    The function that implements and trains the whole algorithm.
    
    Input:
      - data: The loaded dataset
      - par: A dictionary that contains the hyperparameters of the model
      - mod: A string "AR" or "VAR" that selects the model to be used
    
    Returns:
      - Us: The list containing the U matrices
      - convergences: A numpy matrix containing the convergence value of each iteration
      - metric: A numpy matrix containing the rmse and nrmse values of each iteration
      - A: The coefficient matrix
      - prediction: The predicted values of the next step
    """
    
    # Initializations
    conv = 10
    epoch = 0
    convergences = list([])
    metrics = list([])   
        
    # Apply Hankelization
    X_hat, S_pinv = MDT(data_train, par['r'])
    #print("X_hat shape: ", X_hat.shape)
    
    # Differencing
    if par['d']>0:
        X_hat, inv = difference(X_hat, par['d'])

    
    #Rs = get_ranks(X_hat)
    #print("Tucker Ranks: ", Rs)
    Rs = np.array([par['R1'], par['R2']])
    
    # Us Initialization
    Us = initialize_Us(X_hat, Rs) 
    
    # Train the AR model
    while epoch < par['max_epoch'] and conv > par['threshold']:
        old_Us = copy.deepcopy(Us)
        # Step 9 - Calculate the Core Tensors
        G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
        #print("G shape: ", G.shape)
        
        # Calculate par of AR model
        A = fit_model(G, par['p'], mod)       
        
        for m in range(len(Us)):
            # Step 12 - Update cores over m-unfolding
            G = update_cores(m, par['p'], A, Us, X_hat, G, par['lam'], mod)
            # Step 13 - Update U[m]
            Us[m] = update_Um(m, par['p'], X_hat, G, Us)
        
        conv = compute_convergence(Us, old_Us)
        convergences.append(conv)
        epoch += 1
        
        # Re-calculate the core tensor using the updated U matrices
        G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
        
        # Estimate the coefficients for the new core tensor
        A = fit_model(G, par['p'], mod)
        
        # Forecast the next cores
        G_pred = forecast(G, par['p'], A, mod, data_val.shape[-1])
        
        
        dim_list = [u.shape[0] for u in Us]
        dim_list.append(len(G_pred))
        X_pred = np.zeros(tuple(dim_list))
        for i in range(len(G_pred)):
            X_pred[..., i] = tl.tenalg.multi_mode_dot(G_pred[i], Us)

        X_hat_temp = np.append(X_hat, X_pred, axis = -1)
        
        # Inverse Differencing
        if par['d']>0:
            X_hat_temp = inv_difference(X_hat_temp, inv, par['d'])


        pred_mat = tl.tenalg.mode_dot( tl.unfold(X_hat_temp[..., X_pred.shape[-1]:], 0), S_pinv, -1)
        prediction = pred_mat[..., -X_pred.shape[-1]:]
        

        rmse = compute_rmse(prediction, data_val)
        nrmse = compute_nrmse(prediction, data_val)
        metrics.append([rmse, nrmse])
        
    return np.array(convergences), np.array(metrics), A, prediction, Us




def BHTAR_test(data_test_start, data_test, A, Us, par, mod):
    """
    The function that implements and trains the whole algorithm.
    
    Input:
      - data: The loaded dataset
      - par: A dictionary that contains the hyperparameters of the model
      - mod: A string "AR" or "VAR" that selects the model to be used
    
    Returns:
      - Us: The list containing the U matrices
      - convergences: A numpy matrix containing the convergence value of each iteration
      - metric: A numpy matrix containing the rmse and nrmse values of each iteration
      - A: The coefficient matrix
      - prediction: The predicted values of the next step
    """
    X_hat, S_pinv = MDT(data_test_start, par['r'])
    # print("Test X_hat: ", X_hat.shape)
    
    # Differencing
    if par['d']>0:
        X_hat, inv = difference(X_hat, par['d'])
    
    G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)

    G_pred = forecast(G, par['p'], A, mod, data_test.shape[-1])

    dim_list = [u.shape[0] for u in Us]
    dim_list.append(len(G_pred))
    X_pred = np.zeros(tuple(dim_list))
    for i in range(len(G_pred)):
        X_pred[..., i] = tl.tenalg.multi_mode_dot(G_pred[i], Us)

    X_hat_temp = np.append(X_hat, X_pred, axis = -1)

    # Inverse Differencing
    if par['d']>0:
        X_hat_temp = inv_difference(X_hat_temp, inv, par['d'])
    
    pred_mat = tl.tenalg.mode_dot( tl.unfold(X_hat_temp[..., X_pred.shape[-1]:], 0), S_pinv, -1)
    prediction = pred_mat[..., -X_pred.shape[-1]:]
      
    rmse = compute_rmse(prediction, data_test)
    nrmse = compute_nrmse(prediction, data_test)
    
    return rmse, nrmse


