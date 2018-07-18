import numpy as np
import tensorflow as tf
from network import NeuralNetwork
from qutip import fidelity
from qutip.states import coherent, basis
from quoptics.states import cat, squeezed
from quoptics.states import StateIterator

sess = tf.Session()
net = NeuralNetwork(sess)
net.restore("weights")
T = 100

def complex_range(max_x, npts):
    """
    Returns an array consisting of equally spaced points in the square
    of side length max_x centred at the origin in the complex plane.

    :param npts: The number of points in the array. Should be a square number
    :param max_x: Side length of the square to sample
    """
    length = int(np.round(np.sqrt(npts)))
    x_vals = np.linspace(0, max_x, length)
    X, Y = np.meshgrid(x_vals, x_vals)
    X = X.reshape(length**2)
    Y = Y.reshape(length**2)
    output = X + 1j*Y
    return output

FOCK_SPACE = [basis(T, n) for n in np.linspace(0, T-1, T, dtype=int)]
COHERENT_SPACE = [coherent(T, alpha) for alpha in complex_range(1.5, 100)]
SQUEEZED_SPACE = [squeezed(T, z) for z in complex_range(1.7, 100)]
alpha = complex_range(1.0, 50)
theta = np.linspace(0, 2*np.pi, 50)
alpha, theta = np.meshgrid(alpha, theta)
alpha = alpha.reshape(alpha.size)
theta = theta.reshape(theta.size)
CAT_SPACE = [cat(T, *params) for params in zip(alpha, theta)]
print("Search spaces initialized")

def calc_fidelity(state):
    """
    :param state: A qutip.Qobj instance
    """
    input = state.data.toarray().T[0]
    input = np.abs(input)

    type = net.classify(input)

    search_space = None
    if type == 0:
        search_space = FOCK_SPACE
    elif type == 1:
        search_space = COHERENT_SPACE
    elif type == 2:
        search_space = SQUEEZED_SPACE
    elif type == 3:
        search_space = CAT_SPACE
    else:
        raise ValueError(("NeuralNetwork returned unkown classification. "
        "Received {} expected one of{0,1,2,3}").format(type))

    fid_fn = np.vectorize(lambda s: fidelity(state, s))
    fid_values = fid_fn(np.array(search_space))
    return np.max(fid_values)

if __name__ == '__main__':
    # Generate 100 random states (with labels)
    states = [x for x in StateIterator(100, T=100)]
    # Discard the state labels - we don't need them
    states, _ = zip(*states)
    # Calculate the classification probability and fidelity for each state
    probabilities = np.empty(100)
    fidelities = np.empty(100)
    for i, state in enumerate(states):
        data = np.abs(state.data.toarray().T[0])
        probabilities[i] = np.max(net.classify_dist(data))
        fidelities[i] = calc_fidelity(state)
        print("State {}: ({}, {})".format(i, probabilities[i], fidelities[i]))
    # Plot fidelity against probability
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(fidelities, probabilities, s=10, marker='x')
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])
    plt.xlabel("Fidelity")
    plt.ylabel("Probability")
    plt.show()
