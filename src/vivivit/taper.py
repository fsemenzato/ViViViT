import numpy as np


def tukey_window(n, alpha):
    m = np.arange(n, dtype=np.float64)
    if alpha <= 0:
        return np.ones(n, dtype=np.float64)
    if alpha >= 1:
        return 0.5*(1.0-np.cos(2.0*np.pi*m/(n-1)))
    w = np.ones(n, dtype=np.float64)
    edge = alpha*(n-1)/2.0
    lo = m < edge
    hi = m > (n-1)-edge
    w[lo] = 0.5*(1.0+np.cos(np.pi*(2.0*m[lo]/(alpha*(n-1))-1.0)))
    w[hi] = 0.5*(1.0+np.cos(np.pi*(2.0*m[hi]/(alpha*(n-1))-2.0/alpha+1.0)))
    return w


def taper_field(field, alpha, axis=0):
    f = np.asarray(field, dtype=np.float64)
    f = f-f.mean()
    if alpha <= 0:
        return f, 1.0, 1.0
    w = tukey_window(f.shape[axis], alpha)
    shape = [1]*f.ndim
    shape[axis] = -1
    w = w.reshape(shape)
    weighted_mean = (f*w).sum()/(w.sum()*f.size/f.shape[axis])
    return (f-weighted_mean)*w, float(np.mean(w**2)), float(np.mean(w**3))
