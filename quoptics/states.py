"""
Methods for generating different types of state
"""
from qutip.states import coherent, basis
from qutip.operators import squeeze, position
import numpy as np

TYPES = ['fock', 'cat', 'zombie', 'squeezed_cat', 'cubic_phase', 'on']

def cat(T, alpha, theta=0):
    r"""
    Generates a  normalised cat state of the form

    .. math::

        \lvert \text{cat} \rangle_{\theta} = \mathcal{N} \left ( \lvert \alpha \
        \rangle + e^{i\theta} \lvert -\alpha \rangle \right )

    Where :math:`\lvert \alpha \rangle` are :func:`~qutip.states.coherent`
    states

    :param T: The truncation to use
    :param alpha: The complex number prametrising the coherent states
    :param theta: The phase differece between the coherent states
    :returns: A :class:`qutip.Qobj` instance
    """
    a = coherent(T, alpha)
    b = np.exp(1j*theta) * coherent(T, -alpha)
    return (a + b).unit()

def zombie(T, alpha):
    r"""
    Generates a normalised zombie cat state of the form

    .. math::

        \lvert \text{zombie} \rangle = \mathcal{N} \left( \lvert \alpha \rangle + \
        \lvert e^{2 \pi i /3} \alpha \rangle + \
        \lvert e^{4 \pi i /3} \alpha \rangle \right )

    Where :math:`\lvert \alpha \rangle` are :func:`~qutip.states.coherent`
    states

    :param T: The truncation to use
    :param alpha: Complex number parametrising the coherent state
    :returns: A :class:`qutip.Qobj` instance
    """
    a = coherent(T, alpha)
    b = coherent(T, np.exp(2j*np.pi/3)*alpha)
    c = coherent(T, np.exp(4j*np.pi/3)*alpha)
    return (a + b + c).unit()

def squeezed(T, z):
    r"""
    Generates a squeezed state

    .. math::
        \lvert z \rangle = \widehat{S}(z) \lvert 0 \rangle

    :param T: The truncation to use
    :param z: The squeezing parameter
    :returns: A :class:`qutip.Qobj` instance
    """
    vac = basis(T, 0)
    S = squeeze(T, z)
    return S * vac

def squeezed_cat(T, alpha, z):
    r"""
    Generates a squeezed cat state. This is done by generating a perfect
    :func:`~quoptics.states.cat` state (i.e. :math:`\theta=0`), then applying
    the single-mode squeezing operator.

    .. math::
        \lvert \psi \rangle = \widehat{S}(z) \lvert \text{cat} \rangle

    :param T: The truncation to use
    :param alpha: The complex number parametrising the cat state
    :param z: The sqeezing parameter
    :returns: A :class:`qutip.Qobj` instance
    """
    c = cat(T, alpha)
    S = squeeze(T, z)
    return (S * c).unit()

def cubic_phase(T, gamma, z):
    r"""
    Generates a finitely squeezed approximation to a cubic phase state

    .. math::

        \lvert \gamma , z \rangle = e^{i \gamma \widehat{q}^3} \widehat{S} (z)\
        \lvert 0 \rangle

    :param T: The truncation to use
    :param gamma: The parameter of the cubic phase operator
    :param z: The parameter of the squeezing operators
    :returns: A :class:`qutip.Qobj` instance
    """
    q = position(T)
    V = (1j*gamma*q**3).expm()
    S = squeeze(T, z)
    vac = basis(T, 0)
    return (V * S * vac)

def on_state(T, n, delta):
    r"""
    Generates a normalised ON state of the form

    .. math::

        \lvert \text{ON} \rangle = \mathcal{N} \left ( \lvert 0 \rangle + \
        \delta \lvert N \rangle \right )

    :param T: The truncation to use
    :param N: The Fock state to take a superposition of the vacuum with
    :param delta: Coefficient of the non-vacuum Fock state
    :returns: A :class:`qutip.Qobj` instance
    """
    O = basis(T, 0)
    N = basis(T, n)
    return (O + delta*N).unit()

class StateIterator(object):
    r"""
    An `iterator <https://docs.python.org/3/glossary.html#term-iterator>`_
    object that generates random states

    :param batch_size: How many states to generate
    :param T: The truncation to when generating each state
    :param cutoff: The length of the generated state vectors
    :param qutip: Whether to return states as :class:`qutip.Qobj` or numpy
        arrays
    """
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
        elif type == 'cubic_phase':
            gamma = np.random.rand() * 0.25
            z = np.random.exponential(2)
            state = cubic_phase(self.T, gamma, z)
        elif type == 'on':
            n = np.random.randint(1, self.cutoff)
            delta = np.random.rand()
            state = on_state(self.T, n, delta)
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
    r"""
    Returns n randomly generated states and their labels

    :param T: The truncation to use when generating the states
    :param n: How many states to generate
    :param cutoff: The length of the generated state vectors
    :param qutip: Whether to return states as :class:`qutip.Qobj` or numpy
        arrays
    :returns: A tuple containing an array of states and an array of labels
    """
    data = [x for x in StateIterator(n, T=T, cutoff=cutoff, qutip=qutip)]
    states, labels = zip(*data)
    if qutip:
        states = np.array(states)
    else:
        states = np.array([s[:cutoff] for s in states])
    labels = np.array(labels)
    return states, labels
