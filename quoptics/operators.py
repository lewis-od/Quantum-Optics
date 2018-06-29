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

def number(T):
    a = annihilation(T)
    return np.matmul(a.H, a)

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
    a = np.array(annihilation(T))
    aH = np.conj(a).T
    a2 = np.linalg.matrix_power(a,2)
    aH2 = np.linalg.matrix_power(aH,2)
    # a is real so Hermitian conjugate is just transpose
    S = expm(-(z/2)*aH2 + (np.conj(z)/2)*a2)
    return S
