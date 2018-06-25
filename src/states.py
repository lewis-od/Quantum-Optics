import numpy as np
from scipy.special import factorial
from scipy.linalg import expm
import conf

def annihilation(T):
    """
    Returns a truncated annihilation operator as a TxT matrix
    :param T: Dimension of the matrix
    """
    a = np.matrix(np.zeros([T, T]))
    for i in range(T-1):
        a[i, i+1] = np.sqrt(i+1)
    return a

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
    a = annihilation(T)
    D = expm(alpha*a.H - np.conj(alpha)*a) # Displacement operator
    state = np.matmul(D, fock(0, T=T)) # Act on vacuum state with D(alpha)
    state = np.array(state) # Convert from matrix to array
    return state

def squeezed1(z, T=conf.T):
    """
    Squeezed states (single-mode) from analytic expression in Fock basis
    :param z: Complex number that parametrises the squeezed state
    """
    def c(i):
        n = i/2
        cn = 1.0/np.sqrt(np.cosh(np.abs(z)))*np.sqrt(factorial(2*n))
        cn *= 1.0/factorial(n)
        cn *= (-z/(2.0*np.abs(z)))**n
        cn *= np.tanh(np.abs(z))**n
        return cn
    state = [c(n) for n in range(T)]
    zeros = np.zeros(len(state[1::2]))
    state[1::2] = zeros # Squeezed states only have an even number of photons
    return np.array(state)

def squeezed2(z, T=conf.T):
    """
    Squeezed states (single-mode) from squeezing operator
    :param z: Complex number that parametrises the squeezed state
    """
    a = annihilation(T)
    # Single-mode squeezing operator
    S = expm(-(z/2)*(a.H**2) + (np.conj(z)/2)*(a**2))
    state = np.matmul(S, fock(0, T=T))
    state = np.array(state)
    return state
