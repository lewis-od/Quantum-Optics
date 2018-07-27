from qutip.states import coherent, basis
from qutip.operators import squeeze
import numpy as np

TYPES = ['fock', 'cat', 'zombie', 'squeezed_cat']

def cat(T, alpha, theta=0):
    """
    Returns a  normalised cat state of the form
        |alpha> + e^(i*theta)|-alpha>
    Where |alpha> are coherent states
    """
    a = coherent(T, alpha)
    b = np.exp(1j*theta) * coherent(T, -alpha)
    return (a + b).unit()

def zombie(T, alpha):
    """
    Returns a zombie cat state of the form
        |alpha> + |e^(2πi/3)*alpha> + |e^(4πi/3)*alpha>
    Where |alpha> are coherent states
    """
    a = coherent(T, alpha)
    b = coherent(T, np.exp(2j*np.pi/3)*alpha)
    c = coherent(T, np.exp(4j*np.pi/3)*alpha)
    return (a + b + c).unit()

def squeezed(T, z):
    """Returns a squeezed state"""
    vac = basis(T, 0)
    S = squeeze(T, z)
    return S * vac

def squeezed_cat(T, alpha, z):
    """Returns a squeezed cat state"""
    c = cat(T, alpha)
    S = squeeze(T, z)
    return (S * c).unit()

class StateIterator(object):
    def __init__(self, batch_size, T=100, cutoff=25, qutip=True):
        self.n = 0
        self.batch_size = batch_size
        self.T = T
        self.cutoff = cutoff
        self.qutip = qutip
        self.types = TYPES

    def __iter__(self):
        return self

    def _rand_complex(self, modulus):
        r = np.random.rand() * modulus
        theta = np.random.rand() * np.pi * 2
        z = r * np.exp(1j*theta)
        return z

    def __next__(self):
        label = self.n % len(self.types)
        type = self.types[label]

        if type == 'fock':
            n_photons = np.random.randint(0, self.cutoff)
            state = basis(self.T, n_photons)
        elif type == 'cat':
            # Choose sign of cat state at random
            theta = np.random.rand() * np.pi * 2
            alpha = self._rand_complex(1.0)
            state = cat(self.T, alpha, theta)
        elif type == 'zombie':
            alpha = self._rand_complex(1.0)
            state = zombie(self.T, alpha)
        elif type == 'squeezed_cat':
            alpha = self._rand_complex(1.0)
            z = self._rand_complex(1.0)
            state = squeezed_cat(self.T, alpha, z)
        else:
            raise ValueError("Invalid type supplied")

        if self.n == self.batch_size:
            self.n = 0
            raise StopIteration

        if not self.qutip:
            state = np.abs(state.data.toarray().T[0])

        self.n += 1
        return state, label

def random_states(T, n, cutoff=25, qutip=True):
    """
    Returns n randomly generated states and their labels
    """
    data = [x for x in StateIterator(n, T=T, cutoff=cutoff, qutip=qutip)]
    states, labels = zip(*data)
    if qutip:
        states = np.array(states)
    else:
        states = np.array([s[:cutoff] for s in states])
    labels = np.array(labels)
    return states, labels
