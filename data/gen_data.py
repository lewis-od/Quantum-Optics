import sys
import os
# TODO: There must be a better way to do this?
directory = os.path.abspath(os.path.join(__file__, os.pardir))
parent = os.path.abspath(os.path.join(directory, os.pardir))
sys.path.append(parent)

import argparse
from qutip.states import basis, coherent
from quoptics.states import cat, squeezed
import numpy as np

def rand_complex(modulus):
    """Generates a random complex number with |z| < modulus"""
    r = np.random.rand() * modulus
    theta = np.random.rand() * np.pi * 2
    z = r * np.exp(1j*theta)
    return z

def gen_states(n, cutoff):
    """Generates n random states"""
    # Types of state
    types = ['fock', 'coherent', 'squeezed', 'cat']
    # Array to hold all generated states
    states = np.zeros([n, cutoff], dtype=np.complex64)
    labels = np.zeros(n)
    state = None
    for i in range(n):
        # Cycle through the types of state
        id = i % len(types) # Integer identifying type of state
        labels[i] = id
        type = types[id]
        # Generate the state
        if type == 'fock':
            n_photons = np.random.randint(0, cutoff)
            state = basis(T, n_photons)
        elif type == 'coherent':
            alpha = rand_complex(1.0)
            state = coherent(T, alpha)
        elif type == 'squeezed':
            z = rand_complex(1.0)
            state = squeezed(T, z)
        elif type == 'cat':
            # Choose sign of cat state at random
            theta = np.random.rand() * np.pi * 2
            alpha = rand_complex(1.0)
            state = cat(T, alpha, theta)
        else:
            raise ValueError("Invalid type supplied")
        # Convert column vector to row vector
        data = state.data.toarray().T[0]
        states[i] = data[:cutoff]

    return (states, labels)

def save_data(data, name):
    output = os.path.join(directory, name + ".npz")
    if os.path.isfile(output):
        print("File {} already exists, do you want to overwrite it? (y/n)".format(output))
        res = input()
        while res not in ['y', 'n']:
            print("Invalid input: " + res)
            res = input()
        if res == 'n':
            return
    try:
        np.savez(output, states=data[0], labels=data[1])
        print("Created file " + output)
    except Exception as e:
        print("Unable to create file " + output + ": " + str(e))

def parse_arg(n):
    argument = None
    try:
        argument = int(sys.argv[n])
    except:
        pass
    return argument

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description=("Generate data for training "
        "the neural network"))
    parser.add_argument('--training', type=int, required=False, default=5000,
        help='Number of states to generate for training (default: 5000)')
    parser.add_argument('--test', type=int, required=False, default=2000,
        help='Number of states to generate for testing (default: 2000)')
    parser.add_argument('--validation', type=int, required=False, default=2000,
        help='Number of states to generate for validation (default: 2000)')
    parser.add_argument('--truncation', type=int, required=False, default=100,
        help='Size of matrices to use when calculating states', metavar='T')
    parser.add_argument('--cutoff', type=int, required=False, default=25,
        help='Length of state vectors generated', metavar='LEN')

    # Parse arguments
    params = parser.parse_args()

    # Set truncation
    T = params.truncation

    print("Generating data with params {}".format(params.__dict__))

    # Generate data
    training = gen_states(params.training, params.cutoff)
    test = gen_states(params.test, params.cutoff)
    validation = gen_states(params.validation, params.cutoff)

    save_data(training, "train")
    save_data(test, "test")
    save_data(validation, "validation")
