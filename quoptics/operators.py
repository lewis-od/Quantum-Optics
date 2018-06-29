import numpy as np
from scipy.linalg import expm

def annihilation(T):
    """
    Returns the annihilation operator as a TxT matrix
    """
    a = np.zeros([T, T])
    for i in range(T-1):
        a[i, i+1] = np.sqrt(i+1)
    return a

def creation(T):
    """
    Returns the creation operator as a TxT matrix
    """
    return annihilation(T).T

def number(T):
    """
    Returns the number operator as a TxT matrix
    """
    a = annihilation(T)
    c = creation(T)
    return c @ a

def displacement(alpha, T):
    """
    Returns the displacement operator D(alpha) as a TxT matrix
    """
    a = annihilation(T)
    c = creation(T)
    D = expm(alpha*c - np.conj(alpha)*a)
    return D

def squeezing(z, T):
    """
    Returns the single-mode squeezing operator as a TxT matrix
    """
    a = annihilation(T)
    c = creation(T)
    S = expm(-(z/2)*(c @ c) + (np.conj(z)/2)*(a @ a))
    return S
