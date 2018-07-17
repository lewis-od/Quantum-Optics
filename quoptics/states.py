from qutip.states import coherent, basis
from qutip.operators import squeeze
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

class StateIterator(object):
    def __init__(self, batch_size, T=100):
        self.n = 0
        self.batch_size = batch_size
        self.T = T
        self.modulus = True
        self.types = ['fock', 'squeezed', 'coherent', 'cat']

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
            n_photons = np.random.randint(0, self.T)
            state = basis(self.T, n_photons)
        elif type == 'coherent':
            alpha = self._rand_complex(1.0)
            state = coherent(self.T, alpha)
        elif type == 'squeezed':
            z = self._rand_complex(1.0)
            state = squeezed(self.T, z)
        elif type == 'cat':
            # Choose sign of cat state at random
            theta = np.random.rand() * np.pi * 2
            alpha = self._rand_complex(1.0)
            state = cat(self.T, alpha, theta)
        else:
            raise ValueError("Invalid type supplied")

        if self.n == self.batch_size:
            self.n = 0
            raise StopIteration

        self.n += 1
        return state, label

def random_states(T, n):
    """
    Returns n randomly generated states and their labels
    """
    data = [x for x in StateIterator(n, T=T)]
    states, labels = zip(*data)
    states = np.array(states)
    labels = np.array(labels)
    return states, labels
