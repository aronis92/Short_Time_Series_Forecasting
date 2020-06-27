from functions.utils import create_synthetic_data, my_data2
from BHT_ARIMA.util.utility import get_index
from BHT_ARIMA import BHTARIMA
import numpy as np
import random

if __name__ == "__main__":
    np.random.seed(42)
    # prepare data
    # the data should be arranged as (ITEM, TIME) pattern
    # import traffic dataset
    ori_ts = np.load('input/traffic_40.npy').T
    ori_ts = create_synthetic_data(p = 2, dim = 5, n_samples=100)
    # ori_ts = my_data2(p=2, dim=10, n_samples=100)
    print("shape of data: {}".format(ori_ts.shape))
    print("This dataset have {} series, and each serie have {} time step".format(
        ori_ts.shape[0], ori_ts.shape[1]
    ))

    # parameters setting
    ts = ori_ts[..., :-1] # training data, 
    label = ori_ts[..., -1] # label, take the last time step as label
    p = 2 #3 # p-order
    d = 0 #2 # d-order
    q = 0 # q-order
    
    taus = [ori_ts.shape[0], 3] #[228, 5] # MDT-rank
    Rs = [40, 3] # tucker decomposition ranks
    k =  10 # iterations
    tol = 0.001 # stop criterion
    Us_mode = 4 #6 # orthogonality mode

    # For Test only
    # p = 3
    # d = 2
    # q = 1
    # Rs = [5, 3] # tucker decomposition ranks
    
    # Run program
    # result's shape: (ITEM, TIME+1) ** only one step forecasting **
    model = BHTARIMA(ts, p, d, q, taus, Rs, k, tol, verbose=0, Us_mode=Us_mode)
    result, con_l, A = model.run()
    # new_core, new_X, b_list, tmp_H, tmp_X_unfold, Us_starting, result, con_l, Xss, Xs_before_diff, Xs_after_diff, tmp_cores, Us_tmp, A_tmp, B_tmp, tmp_cores_up, Es_tmp, Es_start2 = model.run()
    pred2 = result[..., -1]

    # print extracted forecasting result and evaluation indexes
    print("forecast result(first 10 series):\n", pred2[:10])

    print("Evaluation index: \n{}".format(get_index(pred2, label)))

del label, p,d,q,k,tol,Rs,model,pred2,ts,taus,Us_mode
# Xss = np.transpose(np.array(Xss[:-1]), (1,2,0))
# del Rs, Us_mode, d, k, model, p, q, result, taus, tol, ts, label, ori_ts, Xss, tmp_cores, Us_starting, b_list, con_l #, A_tmp
# del Xs_before_diff, Xs_after_diff, B_tmp, Es_tmp, Es_start2, Us_tmp, tmp_H, tmp_X_unfold, tmp_cores_up, new_X, new_core
