import numpy as np
from abc import ABC, abstractmethod
from . import conf
from scipy.special import factorial
from . import operators as ops

class _State(ABC):
    """
    Base state object
    """
    def __init__(self, T=None, **kwargs):
        self.T = conf.T if T is None else T
        self.data = np.empty(self.T)
        self.params = kwargs
        super().__init__()
        self._gen_data()

    @abstractmethod
    def _gen_data(self):
        pass

class Fock(_State):
    """
    Basis number states
    :param n: Number of the Fock state
    """
    def __init__(self, n, T=None):
        self.type = 'fock'
        self.n = n
        super().__init__(T=T, n=n)

    def _gen_data(self):
        self.data = _fock(self.n, self.T)

class Coherent(_State):
    """
    Coherent states from analytic expression in Fock basis
    :param alpha: Complex number parametrising the coherent state
    """
    def __init__(self, alpha, T=None):
        self.type = 'coherent'
        self.alpha = alpha
        super().__init__(T=T, alpha=alpha)

    def _gen_data(self):
        self.data = _coherent1(self.alpha, self.T)

class Squeezed(_State):
    """
    Squeezed states (single-mode) from analytic expression in Fock basis
    :param z: Complex number that parametrises the squeezed state
    """
    def __init__(self, z, T=None):
        self.type = 'squeezed'
        self.z = z
        super().__init__(T=T, z=z)

    def _gen_data(self):
        self.data = _squeezed1(self.z, self.T)

## Helper methods for generating state data
def _fock(n, T):
    data = np.zeros(T)
    data[n] = 1
    return data

def _coherent1(alpha, T):
    data = [(alpha**n)/np.sqrt(factorial(n)) for n in range(T)]
    data = np.array(data)
    data = data * np.exp(-(np.abs(alpha)**2)/2)
    return data

def _coherent2(alpha, T):
    """
    Coherent states created from the displacement operator
    :param alpha: Complex number parametrising the coherent state
    """
    D = ops.displacement(alpha, T)
    state = np.matmul(D, fock(0, T=T)) # Act on vacuum state with D(alpha)
    state = np.array(state) # Convert from np.matrix to np.array
    return state

def _squeezed1(z, T):
    if z == 0:
        # S(0) is the identity operator, so S(0)|0> = |0>
        return _fock(0, T)

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

def _squeezed2(z, T):
    """
    Squeezed states (single-mode) from squeezing operator
    :param z: Complex number that parametrises the squeezed state
    """
    # Single-mode squeezing operator
    S = ops.squeezing(z, T)
    state = np.matmul(S, fock(0, T=T))
    return np.array(state)
