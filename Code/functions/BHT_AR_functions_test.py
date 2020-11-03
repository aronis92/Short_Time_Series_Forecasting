####################################################
##                                                ##
##  This file contains various utility functions  ##
##                                                ##
####################################################

from functions.utils import compute_rmse, compute_nrmse, get_ranks
from functions.AR_functions import fit_ar, estimate_matrix_coefficients
from functions.MDT_functions import MDT
from statsmodels.tsa.api import VAR
#import statsmodels
import tensorly as tl
import numpy as np
import copy


# The function that initializes and returns the U matrices
# Input:
#   tensor: The data tensor
#   ranks: The rank of each unfolding of the tensor
# Returns:
#   Us: List containing each U matrix
def initialize_Us(tensor, ranks):
    Us = [] # Empty list that will contain the Us Matrices.
    # Exclude the Time dimension since it is the only dimension not decomposed.
    for i in range(len(tensor.shape) - 1):
        # Generate a random array with shape (tensor_size_on_dim_i, rank_on_dim_i)
        Us.append( np.random.rand(tensor.shape[i], ranks[i]) ) 
    return Us


''' Update Functions '''

# The function that updates and returns the core tensors
# Input:
#   m: The mode along which the update will take place
#   p: AR model order
#   A: Coeffients of the AR model
#   Us: List containing the U matrices
#   X: Data in their original form (before decomposition)
#   G: Previous core tensors
#   lam: Reguralization parameter
#   mod: Model to be used
# Returns:
#   G: New core tensors
def update_cores(m, p, A, Us, X, G, lam, mod):
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
            
            if summary.shape != G[..., 0].shape:
                summary = summary.T
            
            temp_G.append(summary/(1 + lam))
            
    elif mod == "VAR":
        for t in range(p, X.shape[-1]):
            v1 = Us[m].T
            v2 = tl.unfold( X[..., t], m)
            v3 = outer.T
            summary = lam * np.linalg.multi_dot( [v1, v2, v3] )
            
            if summary.shape != G[..., 0].shape:
                summary = summary.T
            
            for i in range(p):
                tmp1 = G[..., t - i - 1].flatten()
                tmp1 = tmp1.reshape((tmp1.shape[0], 1))
                tmp2 = np.dot(A[i], tmp1)
                tmp3 = tmp2.reshape((G.shape[0], G.shape[1]))
                summary += tmp3
            
            temp_G.append(summary/(1 + lam))
    
    # if m == 1:
    #     temp_Gd = [tmp.T for tmp in temp_Gd]

    temp_G = np.transpose(np.array(temp_G), (1,2,0))
    n_updatable = temp_G.shape[-1]
    n_all = G.shape[-1]
    G[..., (n_all - n_updatable):] = temp_G
    return G


# The function that updates and returns the U matrix over mode m
# Input:
#   m: The mode along which the update will take place
#   p: AR model order
#   X: Data in their original form (before decomposition)
#   G: Core tensors
#   Us: List containing the U matrices
# Returns:
#   new_Um: New U matrix along mode-m
def update_Um(m, p, X, G, Us):
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


''' Metric Calculation Functions '''

# The function that computes and returns the convergence criterion value
# Input:
#   new_Us: The new U values
#   old_Us: The old U values
# Returns:
#   convergence_value: The value to be checked with the convergence criterion
def compute_convergence(new_Us, old_Us):
    Us_difference = [ new - old for new, old in zip(new_Us, old_Us)]
    a = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in Us_difference], axis=0)
    b = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_Us], axis=0)
    convergence_value = a/b
    return convergence_value


''' Training and Prediction functions '''

# The function that estimates the coefficients of the AR model.
# Input:
#   data: The loaded dataset
#   p: The order of the AR model
#   mod: A string "AR" or "VAR" that selects the model to be used
# Returns:
#   A: The coefficients (scalar or matrix)
def fit_model(data, p, mod):
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






# The function that implements and trains the whole algorithm.
# Input:
#   data: The loaded dataset
#   par: A dictionary that contains the hyperparameters of the model
#   mod: A string "AR" or "VAR" that selects the model to be used
# Returns:
#   Us: The list containing the U matrices
#   convergences: A numpy matrix containing the convergence value of each iteration
#   metric: A numpy matrix containing the rmse and nrmse values of each iteration
#   A: The coefficient matrix
#   prediction: The predicted values of the next step
def BHTAR(data, par, mod):
    
    # Initializations
    conv = 10
    epoch = 0
    convergences = list([])
    metrics = list([])
    
    X_train = data[..., :-par['n_val']]
    X_test = data[..., -par['n_val']]
    
    # Apply Hankelization
    X_hat, S_pinv = MDT(X_train, par['r']) 
    #print("X_hat shape: ", X_hat.shape)
    
    Rs = get_ranks(X_hat)
    print("Tucker Ranks: ", Rs)
    # print(X_hat.shape)
    # Rs = np.array([40, 2])
    Rs = np.array([par['R1'], par['R2']])
    # Rs = np.array([40, 5])
    
    # Us Initialization
    Us = initialize_Us(X_hat, Rs) 
    
    # Train the AR model
    while epoch < par['max_epoch'] and conv > par['threshold']:
        old_Us = copy.deepcopy(Us)
        # Step 9 - Calculate the Core Tensors
        G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
        
        # Calculate par of AR model
        A = fit_model(G, par['p'], mod)       
        
        for m in range(len(Us)):
            # Step 12 - Update cores over m-unfolding
            G = update_cores(m, par['p'], A, Us, X_hat, G, par['lam'], mod)
            # Step 13 - Update U[m]
            Us[m] = update_Um(m, par['p'], X_hat, G, Us)
        
        # print(Us[1])
        conv = compute_convergence(Us, old_Us)
        # print(conv)
        convergences.append(conv)
        epoch += 1

        
        #G_pred_old, A_old = train_predict(X_hat, Us, S_pinv, par, mod)
        
        G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
        A = fit_model(G, par['p'], mod)
        G_pred = forecast(G, par['p'], A, mod)
        

        X_pred = tl.tenalg.multi_mode_dot(G_pred, Us)

        dim_list = list(X_pred.shape)
        dim_list.append(1)
        X_hat_temp = np.append(X_hat, X_pred.reshape( tuple(dim_list) ), axis = -1)
    
        pred_mat = tl.tenalg.mode_dot( tl.unfold(X_hat_temp[..., 1:], 0), S_pinv, -1)
        prediction = pred_mat[:, -1]
        # print(prediction[:5])

        rmse = compute_rmse(prediction, X_test)
        nrmse = compute_nrmse(prediction, X_test)
        metrics.append([rmse, nrmse])
        
    return Us, np.array(convergences), np.array(metrics), A, prediction



def forecast(data, p, A, mod):
    
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
            tmp1 = data[..., -(i + 1)].flatten()
            tmp1 = tmp1.reshape(tmp1.shape[0], 1)
            tmp2 = np.dot(A[i], tmp1)
            tmp3 = tmp2.reshape(data.shape[0], data.shape[1])
            prediction += tmp3
    
    return prediction



# The function that trains the AR model and returns the next step prediction and the coefficients of the model.
# Input:
#   X_hat: The Hankelized data
#   Us: List that contains all U matrices
#   S_pinv: The pseudoinverse matrix to de-Hankelize the data
#   par: A dictionary that contains the hyperparameters of the model
#   mod: The model that will be used for the coefficient estimation
# Returns:
#   pred_value: The predicted value
#   A: The coefficient matrix
def train_predict(X_hat, Us, S_pinv, par, mod):
    p = par['p']
    # Estimate the core tensor with the given Us.
    G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
    
    A = fit_model(G, par['p'], mod) 
         
    # Predict the core
    G_pred = 0
    if mod == "AR":
        for i in range(p):
            G_pred += A[i] * G[..., -(i + 1)]
        X_pred = tl.tenalg.multi_mode_dot(G_pred, Us)
    
    elif mod == "myVAR":
        G_pred = A[0]
        for i in range(p):
            tmp1 = G[..., -(i + 1)].flatten()
            G_pred += np.dot(A[i + 1], tmp1)
        G_pred = G_pred.reshape(G.shape[0], G.shape[1])
        
    elif mod == "VAR":
        for i in range(p):
            tmp1 = G[..., -(i + 1)].flatten()
            tmp1 = tmp1.reshape(tmp1.shape[0], 1)
            tmp2 = np.dot(A[i], tmp1)
            tmp3 = tmp2.reshape(G.shape[0], G.shape[1])
            G_pred += tmp3
    
    X_pred = tl.tenalg.multi_mode_dot(G_pred, Us)

    dim_list = list(X_pred.shape)
    dim_list.append(1)
    X_hat = np.append(X_hat, X_pred.reshape( tuple(dim_list) ), axis = -1)

    pred_mat = tl.tenalg.mode_dot( tl.unfold(X_hat[..., 1:], 0), S_pinv, -1)
    pred_value = pred_mat[:, -1]
    return G_pred, A
    #return pred_value, A




def predict(mod, Us, A, par, X, X_test):
    p = par['p']
    X_hat, S_pinv = MDT(X, par['r'])
    # Estimate the core tensor with the given Us.
    G = tl.tenalg.multi_mode_dot(X_hat, Us, modes = [i for i in range(len(Us))], transpose = True)
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
            
    elif mod == "myVAR":
        G_pred = A[0]
        print("G_pred shape 1: ", G_pred.shape)
        for i in range(p):
            tmp1 = G[..., -(i + 1)].flatten()
            #tmp1 = tmp1.reshape(tmp1.shape[0], 1)
            print("tmp 1 shape: ", tmp1.shape)
            G_pred += np.dot(A[i + 1], tmp1)
            #print("tmp 2 shape: ", tmp2.shape)
            G_pred = G_pred.reshape(G.shape[0], G.shape[1])
    
    X_pred = tl.tenalg.multi_mode_dot(G_pred, Us)
            
    dim_list = list(X_pred.shape)
    dim_list.append(1)
    X_hat = np.append(X_hat, X_pred.reshape( tuple(dim_list) ), axis = -1)
    # Reverse differencing should happen here
    pred_mat = tl.tenalg.mode_dot( tl.unfold(X_hat[..., 1:], 0), S_pinv, -1)
    pred_value = pred_mat[:, -1]
    pred_value = pred_value.reshape(pred_value.shape[0], 1)
    
    rmse = compute_rmse(pred_value, X_test)
    nrmse = compute_nrmse(pred_value, X_test)
    return rmse, nrmse








