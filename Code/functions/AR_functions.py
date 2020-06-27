import numpy as np
import scipy as sp

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
    r = r[1:]
    A = sp.linalg.pinv(R).dot(r)
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
        Res = np.transpose(Res, (1, 2, 0))
        B = fit_ar(Res, q)
    return A, B