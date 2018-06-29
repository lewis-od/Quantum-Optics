import numpy as np
from abc import ABC, abstractmethod
from . import conf
from scipy.special import factorial
from . import operators as ops

class _State(ABC):
    """
    Base state object
    """
    def __init__(self, T=None):
        self.T = conf.T if T is None else T
        self.data = np.empty(self.T)
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
        super().__init__(T=T)

    def _gen_data(self):
        # Construct Fock state
        self.data = np.zeros(self.T)
        self.data[self.n] = 1

class Coherent(_State):
    """
    Coherent states from analytic expression in Fock basis
    :param alpha: Complex number parametrising the coherent state
    """
    def __init__(self, alpha, T=None):
        self.type = 'coherent'
        self.alpha = alpha
        super().__init__(T=T)

    def _gen_data(self):
        data = [(self.alpha**n)/np.sqrt(factorial(n)) for n in range(self.T)]
        data = np.array(data)
        data = data * np.exp(-(np.abs(self.alpha)**2)/2)
        self.data = data

class Squeezed(_State):
    """
    Squeezed states (single-mode) from analytic expression in Fock basis
    :param z: Complex number that parametrises the squeezed state
    """
    def __init__(self, z, T=None):
        self.type = 'squeezed'
        self.z = z
        super().__init__(T=T)

    def _gen_data(self):
        if self.z == 0:
            # S(0) is the identity operator, so S(0)|0> = |0>
            vacuum = Fock(0, T=self.T)
            self.data = vacuum.data
            return

        def c(i):
            """Coefficient of basis state |i>"""
            n = i/2
            cn = 1.0/np.sqrt(np.cosh(np.abs(self.z)))*np.sqrt(factorial(2*n))
            cn *= 1.0/factorial(n)
            cn *= (-self.z/(2.0*np.abs(self.z)))**n
            cn *= np.tanh(np.abs(self.z))**n
            return cn
        state = [c(n) for n in range(self.T)]

        # Squeezed states only have an even number of photons - set coefficients of
        # odd Fock states to 0
        zeros = np.zeros(len(state[1::2]))
        state[1::2] = zeros
        self.data = np.array(state)

def coherent2(alpha, T=None):
    """
    Coherent states created from the displacement operator
    :param alpha: Complex number parametrising the coherent state
    """
    if T is None: T = conf.T
    D = ops.displacement(alpha, T)
    state = np.matmul(D, fock(0, T=T)) # Act on vacuum state with D(alpha)
    state = np.array(state) # Convert from np.matrix to np.array
    return state

def squeezed2(z, T=None):
    """
    Squeezed states (single-mode) from squeezing operator
    :param z: Complex number that parametrises the squeezed state
    """
    if T is None: T = conf.T
    # Single-mode squeezing operator
    S = ops.squeezing(z, T)
    state = np.matmul(S, fock(0, T=T))
    return np.array(state)
