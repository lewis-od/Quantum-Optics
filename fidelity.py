import os
import datetime
import numpy as np
from network import NeuralNetwork
from qutip import fidelity
from qutip.states import coherent, basis
from quoptics.states import cat, squeezed_cat, zombie, cubic_phase, on_state, random_states

cur_dir = os.path.abspath('.')
model_dir = os.path.join(cur_dir, "model")
net = NeuralNetwork(model_dir=model_dir)
T = 25

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

def convert_to_numpy(quobjs):
    return [s.data.toarray().T[0] for s in quobjs]

start_time = datetime.datetime.now()
# Create the search space an an array of states generated by QuTiP
FOCK_SPACE = [basis(T, n) for n in np.linspace(0, T-1, T, dtype=int)]

alpha = complex_range(1.0, 50)
theta = np.linspace(0, 2*np.pi, 50)
alpha, theta = np.meshgrid(alpha, theta)
alpha = alpha.reshape(alpha.size)
theta = theta.reshape(theta.size)
CAT_SPACE = [cat(T, *params) for params in zip(alpha, theta)]

ZOMBIE_SPACE = [zombie(T, alpha) for alpha in complex_range(1.0, 100)]

alpha = complex_range(1.0, 50)
alpha, z = np.meshgrid(alpha, alpha)
alpha = alpha.reshape(alpha.size)
z = z.reshape(z.size)
SQUEEZED_CAT_SPACE = [squeezed_cat(T, *params) for params in zip(alpha, z)]

ns = np.arange(0, T)
deltas = np.linspace(0, 1.0, 100)
ns, deltas = np.meshgrid(ns, deltas)
ns = ns.reshape(ns.size)
deltas = deltas.reshape(deltas.size)
ON_SPACE = [on_state(T, *params) for params in zip(ns, deltas)]

# Convert QuTiP Qobjs to numpy arrays
FOCK_SPACE = convert_to_numpy(FOCK_SPACE)
CAT_SPACE = convert_to_numpy(CAT_SPACE)
ZOMBIE_SPACE = convert_to_numpy(ZOMBIE_SPACE)
SQUEEZED_CAT_SPACE = convert_to_numpy(SQUEEZED_CAT_SPACE)
ON_SPACE = convert_to_numpy(ON_SPACE)
end_time = datetime.datetime.now()
print("Search spaces initialized - " + str(end_time - start_time))

def calc_fidelity(state_info):
    """
    Calculates the fidelity of the supplied state with each state in the
    search space, and returns the maximum value

    :param state_info: A tuple containing the state vector, and an integer
    indicating the type of state
    """
    state, type = state_info
    search_space = None
    if type == 0:
        search_space = FOCK_SPACE
    elif type == 1:
        search_space = CAT_SPACE
    elif type == 2:
        search_space = ZOMBIE_SPACE
    elif type == 3:
        search_space = SQUEEZED_CAT_SPACE
    elif type == 4:
        return None
    elif type == 5:
        search_space = ON_SPACE
    else:
        raise ValueError(("NeuralNetwork returned unkown classification. "
        "Received {} expected one of{0,1,2,3}").format(type))

    # Calculate the fidelity with each state in the search space
    fid_values = [np.abs(np.dot(np.conj(state), x)) for x in search_space]
    return np.max(fid_values)

if __name__ == '__main__':
    # Generate 100 random states (discard labels)
    state_data, _ = random_states(T, 100, qutip=False)

    # Classify the states and calculate the corresponding classification prob.
    start_time = datetime.datetime.now()
    predictions = net.predict(state_data)
    probabilities = np.array([np.max(p['probabilities']) for p in predictions])
    classifications = np.array([p['class_ids'][0] for p in predictions])
    end_time = datetime.datetime.now()
    print("Probabilities calculated - " + str(end_time - start_time))

    # Calculate the fidelity for each state
    start_time = end_time
    fidelities = [calc_fidelity(info) for info in zip(state_data, classifications)]
    end_time = datetime.datetime.now()
    print("Fidelities calculated - " + str(end_time - start_time))

    import pdb; pdb.set_trace()
    fidelities = np.array(fidelities)
    probabilities = np.array(probabilities)

    probabilities = probabilities[fidelities != None]
    fidelities = fidelities[fidelities != None]
    # Plot fidelity against probability
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(fidelities, probabilities, s=10, marker='x')
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])
    plt.xlabel("Fidelity")
    plt.ylabel("Probability")
    plt.show()
