import numpy as np
from scipy.linalg import expm

def annihilation(T):
    """
    Returns a truncated annihilation operator as a TxT matrix
    :param T: Dimension of the matrix
    """
    a = np.matrix(np.zeros([T, T]))
    for i in range(T-1):
        a[i, i+1] = np.sqrt(i+1)
    return a

def displacement(alpha, T):
    """
    Returns the displacement operator D(alpha) as a TxT matrix
    """
    a = annihilation(T)
    D = expm(alpha*a.H - np.conj(alpha)*a)
    return D

def squeezing(z, T):
    """
    Returns the single-mode squeezing operator as a TxT matrix
    """
    a = annihilation(T)
    S = expm(-(z/2)*(a.H**2) + (np.conj(z)/2)*(a**2))
    return S   
