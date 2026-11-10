import numpy as np

def a_of_M95(M95):
    return float(0.05 ** (1.0 / (2.0 * M95)))

def _filter(S, k0):
    ki = np.fft.fftfreq(S) * S
    kx = np.minimum(np.abs(ki), S - np.abs(ki))
    (kX, kY) = np.meshgrid(kx, kx, indexing='ij')
    kperp = np.sqrt(kX ** 2 + kY ** 2)
    F = np.exp(-0.5 * (kperp / k0) ** 4)
    F[0, 0] = 0.0
    return F

def sample_forcing(T, S, rng, k0=8.0):
    F = _filter(S, k0)
    norm = np.sqrt((F ** 2).sum() / (S * S))
    out = np.empty((T, S, S), dtype=np.float32)
    for t in range(T):
        w = rng.standard_normal((S, S))
        f = np.real(np.fft.ifft2(np.fft.fft2(w) * F))
        out[t] = (f / (norm + 1e-12)).astype(np.float32)
    return out

def coupled_ou(eps, a, b=0.8):
    T = eps.shape[0]
    (sa, sb) = (np.sqrt(1 - a * a), np.sqrt(1 - b * b))
    A = np.empty_like(eps)
    B = np.empty_like(eps)
    A[0] = eps[0]
    B[0] = eps[0]
    for t in range(1, T):
        A[t] = b * A[t - 1] + sb * eps[t]
        B[t] = a * B[t - 1] + sa * eps[t]
    return (A.astype(np.float32), B.astype(np.float32))
