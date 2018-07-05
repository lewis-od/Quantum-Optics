import sys
import os
# TODO: There must be a better way to do this?
directory = os.path.abspath(os.path.join(__file__, os.pardir))
parent = os.path.abspath(os.path.join(directory, os.pardir))
sys.path.append(parent)

import argparse
import numpy as np
import quoptics as qo

def rand_complex(modulus):
    """Generates a random complex number with |z| < modulus"""
    r = np.random.rand() * modulus
    theta = np.random.rand() * np.pi * 2
    z = r * np.exp(1j*theta)
    return z

def gen_states(n):
    """Generates n random states"""
    # Types of state
    types = ['fock', 'coherent', 'squeezed', 'cat']
    fock = qo.states.Fock(0)
    coherent = qo.states.Coherent(0)
    squeezed = qo.states.Squeezed(0)
    cat = qo.states.Cat(0)
    # Array to hold all generated states
    states = np.zeros([n, qo.conf.T], dtype=np.complex64)
    labels = np.zeros(n)
    for i in range(n):
        # Cycle through the types of state
        id = i % len(types) # Integer identifying type of state
        labels[i] = id
        type = types[id]
        # Generate the state
        if type == 'fock':
            fock.n = np.random.randint(0, qo.conf.T)
            states[i] = fock.data
        elif type == 'coherent':
            coherent.alpha = rand_complex(1.0)
            states[i] = coherent.data
        elif type == 'squeezed':
            squeezed.z = rand_complex(1.0)
            states[i] = squeezed.data
        elif type == 'cat':
            # Choose sign of cat state at random
            cat.parity = ['+', '-'][np.random.randint(2)]
            cat.alpha = rand_complex(1.0)
            states[i] = cat.data
        else:
            raise ValueError("Invalid type supplied")

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
        help='Length of state vectors generated (default: 100)')

    # Parse arguments
    params = parser.parse_args()

    # Set module-wide truncation
    qo.conf.T = params.truncation

    print("Generating data with params {}".format(params.__dict__))

    # Generate data
    training = gen_states(params.training)
    test = gen_states(params.test)
    validation = gen_states(params.validation)

    save_data(training, "train")
    save_data(test, "test")
    save_data(validation, "validation")
