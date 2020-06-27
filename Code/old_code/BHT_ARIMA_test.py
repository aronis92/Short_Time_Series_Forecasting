import tensorly as tl
import numpy as np
import scipy as sp
import copy
import random
import matplotlib.pyplot as plt
from tensorly.tenalg.proximal import procrustes
# from sklearn.metrics import mean_squared_error
from math import sqrt

def load_data(s):
    if s == 2:
        return np.load('traffic_40.npy').T
    elif s == 3:
        X = np.array([i for i in range(6*5*4)])
        return X.reshape((6,5,4))

def create_synthetic_data(p, q, d, dim, n_samples, mu, sigma):
    np.random.seed(0)
    random.seed(0)

    X = np.random.random_sample((dim, p))
    E = np.random.normal(mu, sigma, (n_samples + q, dim, q + 1))
    A = np.array([0.4563476, 0.5591154]).reshape((p, 1))
    B = np.array([-1.111807, 0]).reshape((q, 1))

    for t in range( n_samples):
        x_t = np.dot(X[:, t:(t + p)], A) - np.dot(E[t, :, :-1], B) + E[t, :, -1].reshape(dim, 1)
        X = np.hstack((X, x_t))
    
    X2 = X[:, p + d:] + X[:, p:-d]
    return X[:, -n_samples:], X2

#X, X_new = create_synthetic_data(p=2, q=2, d=1, dim=1, n_samples=100, mu=0, sigma=1)

def get_duplication_matrix(r, T):
    S = np.vstack((np.eye(r), np.zeros((T-r, r))))
    for i in range(T- r):
        end = (i + 1)*r  
        temp1 = S[:-1, (end - r):end]
        temp2 = np.vstack((np.zeros((1, r)), temp1))
        S = np.hstack((S, temp2))
    return S.T   

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
    return np.transpose(X, (1, 2, 0)), S

def differencing(data, d):
    new_data = []
    for i in range(d, data.shape[-1]):
        new_data.append( data[..., i] - data[..., i - d] )
    new_data = np.array(new_data)
    shape = [i for i in range(1, len(new_data.shape))]
    shape.append(0)
    return np.transpose(new_data, tuple(shape))

def get_ranks(tensor):
    ranks = []
    for i in range(len(tensor.shape) - 1):
        temp = tl.unfold(tensor, i)
        ranks.append( np.linalg.matrix_rank(temp) )
    return np.array(ranks)

def initialize_Us(Xd, ranks):
    Us = []
    for i in range(len(Xd.shape) - 1):
        Us.append( np.random.rand(Xd.shape[i], ranks[i]) )
    return Us

def initialize_Es(ranks, p, q, T_hat, s):
    r = list(ranks)
    #r.append(T_hat - s + 1)
    r.append(T_hat - s)
    r = tuple(r)
    return np.random.random_sample(r)

def autocorrelation(data, p):
    T = data.shape[-1]
    r = []
    for l in range(p + 1):
        product = 0
        for t in range(T):
            tl = l - t if t < l else t - l
            product += np.sum(data[..., t] * data[..., tl])
        r.append(product)
    return r

def fit_ar(data, p):
    r = autocorrelation(data, p)
    R = sp.linalg.toeplitz(r[:p])
    A = np.dot(sp.linalg.pinv(R), r[1:])
    return A

def fit_ar_ma(data, p, q):
    N = data.shape[-1]
    A = fit_ar(data, p)
    B = [0.]
    if q > 0:
        Res = []
        for i in range(p, N):
            res = data[..., i] - np.sum([ a * data[..., i-j] for a, j in zip(A, range(1, p + 1))], axis=0)
            Res.append(res)
        Res = np.array(Res)
        B = fit_ar(Res, q)
    return A, B

def update_cores(m, p, q, s, Us, Xd, Gd, Es, lam, A, B):
    temp_Gd = []
    temp_Us = copy.deepcopy(Us) # Copy all Us
    _ = temp_Us.pop(m) # Remove the one in position m to calculate the outer product of the others
    outer = temp_Us[0].T # Starting point of outer product
    
    # Calculate the outer product of Us_minus_m
    for k in range(1, len(temp_Us)):
        outer = np.outer(outer, temp_Us[k].T)
    
    for t in range(s, Xd.shape[-1]):
        v1 = Us[m].T
        v2 = tl.unfold( Xd[..., t], m)
        v3 = outer.T
        summary = lam * np.linalg.multi_dot( [v1, v2, v3] )
        
        for i in range(p):
            summary += A[i] * tl.unfold(Gd[..., t - i - 1], m)
            
        for j in range(q):
            summary -= B[j] * tl.unfold(Es[..., t - j - s ], m)
        
        if summary.shape != Gd[..., 0].shape:
            summary = summary.T
        
        temp_Gd.append(summary/(1 + lam))

    temp_Gd = np.transpose(np.array(temp_Gd), (1,2,0))
    n_updatable = temp_Gd.shape[-1]
    n_all = Gd.shape[-1]
    Gd[..., (n_all - n_updatable):] = temp_Gd
    return Gd

def update_Um(m, p, d, q, Xd, Gd):
    summary = 0
    for t in range(p + q + d, Xd.shape[-1]):
        v1 = tl.unfold(Xd[..., t], m)
        v2 = Us[abs(m - 1)] 
        v3 = tl.unfold(Gd[..., t], m).T
        summary += np.linalg.multi_dot([v1, v2, v3])
    
    return tl.tenalg.proximal.procrustes(summary)

def update_Es(m, p, q, s, Gd, Es, A, B):
    for outer_i in range(q):
        length = Gd.shape[-1] - s
        summary = tl.unfold(Gd[..., -length:], m)
        print("Summary length: ", summary.shape)
        for i in range(1, p + 1):
            print("Shape :",  tl.unfold(Gd[..., -(length + i):(-i)], m).shape)
            summary -= A[i - 1] * tl.unfold(Gd[..., -(length + i):(-i)], m)
                
        js = [_ for _ in range(q)]
        _ = js.pop(outer_i)
        for j in js:
            print("Shape :",  tl.unfold(Es[..., -(length + j + 1):(-j - 1)], m).shape)
            summary += B[j] * tl.unfold(Es[..., -(length + j + 1):(-j - 1)], m)
            
        summary = tl.fold(summary, m, Gd[..., -length:].shape)
        
        print(summary.shape)
        # print((s + 1 - Gd.shape[-1] * B[outer_i]))
        Es[..., outer_i, -length:] = summary / (s + 1 - Gd.shape[-1] * B[outer_i])
    return Es

def update_Es2(m, p, q, s, Gd, Es, A, B):
    for outer_i in range(q):
        
        length = Gd.shape[-1] - s
        summary = tl.unfold(Gd[..., -length:], m)
        # print("Summary length: ", summary.shape)
        for i in range(1, p + 1):
            # print("Shape :",  tl.unfold(Gd[..., -(length + i):(-i)], m).shape)
            summary -= A[i - 1] * tl.unfold(Gd[..., -(length + i):(-i)], m)
                
        js = [_ for _ in range(1, q + 1)]
        _ = js.pop(outer_i)
        for j in js:
            # print("Shape :",  tl.unfold(Es[..., -(length + j + 1):(-j - 1)], m).shape)
            summary += B[j] * tl.unfold(Es[..., -(length + j):(-j)], m)
            
        summary = tl.fold(summary, m, Gd[..., -length:].shape)
        
        print(summary.shape)
        # print((s + 1 - Gd.shape[-1] * B[outer_i]))
        Es[..., outer_i, -length:] = summary / (s + 1 - Gd.shape[-1] * B[outer_i])
    return Es

def predict(p, q, d, Gd, Es, Us, X_hat, A, B):
    
    # Predict the core G for the T_hat + 1 timestep
    Gd_pred = 0
    for i in range(p):
        Gd_pred += A[i] * Gd[..., -(i + 1)]
    for i in range(q):
        Gd_pred -= B[i] * Es[..., i, -(i + 1)]
        
    # Convert the core to the original X for the T_hat + 1 timestep
    Xd_pred = tl.tenalg.multi_mode_dot(Gd_pred, Us, modes = [i for i in range(len(Us))], transpose = False)
    
    # Construct the new X_hat that contains the T_hat most recent timesteps including the predicted Xd_pred
    X_hat_new = np.empty( tuple( list(X_hat.shape) ) )
    X_hat_new[..., :-1] = X_hat[..., 1:]
    X_hat_new[..., -1] = Xd_pred + X_hat[..., -p]
    
    predictions = tl.tenalg.mode_dot( tl.unfold(X_hat_new, 0), S_pseudo_inverse, -1)
    return predictions[..., -1]

def plot_results(data, title):
    epoch = [i + 1 for i in range(len(data))]
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.plot(epoch[1:], data[1:])
    #plt.savefig(title + "2")
    

# Load Dataset
X = load_data(2)
#X, X_new = create_synthetic_data(p=2, q=2, d=1, dim=228, n_samples=40, mu=0, sigma=1)
# Set some variables
r = 4
p = 3
d = 0
q = 2
s = p + d + q
lam = 1
X_train = X[:,:-1]
X_test = X[:,-1]
T = X_train.shape[-1]
T_hat = T - r + 1
max_epoch = 20
conv = 10
epoch = 0
threshold = 0.001
convergences = list([])
pred_diff = list([])

X_hat, S = MDT(X_train, r) # Apply MDT on the training Data
S_pseudo_inverse = np.dot( np.linalg.inv( np.dot(S.T, S) ), S.T )

# Before or After Xd calculation?
ranks = get_ranks(X_hat) # Calculate the rank of each dimension
Us = initialize_Us(X_hat, ranks) # initialize U matrices

# Apply differencing if not stationary
if d>0:
    Xd = differencing(X_hat, d) # Apply Differencing to make stationary
else:
    Xd = X_hat

Es = initialize_Es(ranks, p, q, T_hat, s) # Initialize the random Es

while epoch < max_epoch and conv > threshold:
    
    new_Us = [[],[]]
    denominator = 0
    numerator = 0
    
    # Step 9 - Calculate the Core Tensors
    Gd = tl.tenalg.multi_mode_dot(Xd, Us, modes = [i for i in range(len(Us))], transpose = True)
    
    #Step 10 Calculate a's, b's from Yule Walker equations
    A, B = fit_ar_ma(Gd, p, q)
    # A = np.flip(A)
    # B = np.flip(B)
    
    for m in range(len(Us)):

        # Step 12 - Update cores over m-unfolding
        Gd = update_cores(m, p, q, s, Us, Xd, Gd, Es, lam, A, B)

        # Step 13 - Update U[m]
        new_Us[m] = update_Um(m, p, d, q, Xd, Gd)

        # Step 15, 16 - Update Es
        Es = update_Es2(m, p, q, s, Gd, Es, A, B)

        numerator += np.linalg.norm( new_Us[m] - Us[m], ord = 'fro')**2
        denominator += np.linalg.norm( new_Us[m], ord = 'fro')**2
    
    Us = copy.deepcopy(new_Us)
    conv = numerator/denominator
    convergences.append(conv)
    epoch += 1
    

    pred = predict(p, q, d, Gd, Es, Us, X_hat, A, B)
    
    #pred_diff.append( np.linalg.norm(pred - X_test) / np.linalg.norm(X_test) )
    
    # RMSE
    Res = np.abs(pred - X_test)
    rmse = np.linalg.norm(Res) / sqrt(len(Res))
    #NRMSE = rmse / np.std(Res)
    #NRMSE = rmse / np.mean(Res)
    NRMSE = rmse / (np.max(Res) - np.min(Res))
    
    pred_diff.append( NRMSE )



plot_results(convergences, "Convergence Criterion")
plot_results(pred_diff, "NRMSE")

