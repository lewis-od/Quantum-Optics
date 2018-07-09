from qutip import *
import numpy as np

def cat(T, alpha, theta=0):
    """
    Returns a  normalised cat state of the form
        |alpha> + e^(i*theta)|-alpha>
    Where |alpha> are coherent states
    """
    a = coherent(T, alpha)
    b = np.exp(1j*theta) * coherent(T, -alpha)
    return (a + b).unit()

def squeezed(T, z):
    """Returns a squeezed state"""
    vac = basis(T, 0)
    S = squeeze(T, z)
    return S * vac
