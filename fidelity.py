import numpy as np
import tensorflow as tf
from network import NeuralNetwork
from qutip import fidelity
from qutip.states import coherent, basis
from quoptics.states import cat, squeezed

sess = tf.Session()
net = NeuralNetwork(sess)
net.restore("weights")
T = 100

def complex_range(max_x, npts):
    length = int(np.round(np.sqrt(npts)))
    x_vals = np.linspace(0, max_x, length)
    X, Y = np.meshgrid(x_vals, x_vals)
    X = X.reshape(length**2)
    Y = Y.reshape(length**2)
    output = X + 1j*Y
    return output


def find_fidelity_fn(state):
    """
    Classifies the input state using the neural network, then returns a function
    that calculates the fidelity between the input state and the type of state
    it was classified as, along with a range of parameter values to try.
    For example, if the input state |psi> is classified as a coherent state, the
    function returned will be:
        f(alpha) = |<psi|alpha>|^2
    And trial_values will be a list of complex numbers covering the grid formed
    by +/-1.5 and +/-1.5i

    :param state: A qutip.Qobj instance
    """
    input = state.data.toarray().T[0]
    input = np.abs(input)

    type = net.classify(input)
    trial_values = None

    fid_fn = None
    if type == 0:
        fid_fn = lambda param: fidelity(state, basis(T, param))
        trial_values = np.linspace(0, T-1, T, dtype=int)
    elif type == 1:
        fid_fn = lambda param: fidelity(state, coherent(T, param))
        trial_values = complex_range(1.5, 100)
    elif type == 2:
        fid_fn = lambda param: fidelity(state, squeezed(T, param))
        trial_values = complex_range(1.7, 100)
    elif type == 3:
        fid_fn = lambda param: fidelity(state, cat(T, param))
        trial_values = complex_range(1.0, 100)
    else:
        raise Exception(("Neural network returned unknown classificaiton. "
        "Expected one of [0,1,2,3] but received {}.").format(type))

    return fid_fn, trial_values

def calc_fidelity(state):
    fid, trial = find_fidelity_fn(state)
    fid_values = list(map(fid, trial))
    return np.max(fid_values)
