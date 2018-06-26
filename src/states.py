import numpy as np
import conf
from scipy.special import factorial
import operators as ops

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
    D = ops.displacement(alpha, T)
    state = np.matmul(D, fock(0, T=T)) # Act on vacuum state with D(alpha)
    state = np.array(state) # Convert from np.matrix to np.array
    return state

def squeezed1(z, T=conf.T):
    """
    Squeezed states (single-mode) from analytic expression in Fock basis
    :param z: Complex number that parametrises the squeezed state
    """
    if z == 0:
        # S(0) is the identity operator
        return fock(T=T)

    def c(i):
        """Coefficient of basis state |i>"""
        n = i/2
        cn = 1.0/np.sqrt(np.cosh(np.abs(z)))*np.sqrt(factorial(2*n))
        cn *= 1.0/factorial(n)
        cn *= (-z/(2.0*np.abs(z)))**n
        cn *= np.tanh(np.abs(z))**n
        return cn
    state = [c(n) for n in range(T)]

    # Squeezed states only have an even number of photons - set coefficients of
    # odd Fock states to 0
    zeros = np.zeros(len(state[1::2]))
    state[1::2] = zeros
    return np.array(state)

def squeezed2(z, T=conf.T):
    """
    Squeezed states (single-mode) from squeezing operator
    :param z: Complex number that parametrises the squeezed state
    """
    # Single-mode squeezing operator
    S = ops.squeezing(z, T)
    state = np.matmul(S, fock(0, T=T))
    return np.array(state)
