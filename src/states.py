import numpy as np
from scipy.special import factorial
from scipy.linalg import expm
import conf

def fock(n, T=conf.T):
    """
    Basis number states
    :param n: Number of the Fock state
    """
    # Check n is non-negative int less than T
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer")
    elif n >= T:
        raise ValueError("n must be less than T")
    # Construct Fock state
    f = np.zeros(T)
    f[n] = 1
    return f

def coherent1(alpha, T=conf.T):
    """
    Coherent states from analytic expression in Fock basis
    :param alpha: Complex number parametrising the coherent state
    """
    state = [(alpha**n)/np.sqrt(factorial(n)) for n in range(T)]
    state = np.array(state)
    state = state * np.exp(-(np.abs(alpha)**2)/2)
    return state

def coherent2(alpha, T=conf.T):
    """
    Coherent states created from the displacement operator
    :param alpha: Complex number parametrising the coherent state
    """
    # Annhilation operator as a matrix
    a = np.matrix(np.zeros([T, T]))
    for i in range(T-1):
        a[i, i+1] = np.sqrt(i+1)

    D = expm(alpha*a.H - np.conj(alpha)*a) # Displacement operator
    state = np.matmul(D, fock(0, T=T)) # Act on vacuum state with D(alpha)
    state = np.array(state) # Convert from matrix to array
    return state
