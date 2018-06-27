import sys
import os
# TODO: There must be a better way to do this?
directory = os.path.abspath(os.path.join(__file__, os.pardir))
parent = os.path.abspath(os.path.join(directory, os.pardir))
sys.path.append(parent)

import argparse
import numpy as np
import quoptics as qo

def gen_states(n):
    """Generates n random states"""
    # Types of state
    types = ['fock', 'coherent', 'squeezed']
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
            param = np.random.randint(0, qo.conf.T)
            states[i] = qo.states.fock(param)
        elif type == 'coherent':
            # Random complex num with |alpha| < 1
            r = np.random.rand()
            theta = np.random.rand() * np.pi * 2
            alpha = r * np.exp(1j * theta)
            states[i] = qo.states.coherent(alpha)
        elif type == 'squeezed':
            # Random complex number with |z| < 1
            r = np.random.rand()
            theta = np.random.rand() * np.pi * 2
            z = r * np.exp(1j * theta)
            states[i] = qo.states.squeezed(z)
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
    parser.add_argument('--training', type=int, required=False, default=100,
        help='Number of states to generate for training (default: 100)')
    parser.add_argument('--test', type=int, required=False, default=25,
        help='Number of states to generate for testing (default: 25)')
    parser.add_argument('--validation', type=int, required=False, default=25,
        help='Number of states to generate for validation (default: 25)')
    parser.add_argument('--truncation', type=int, required=False, default=50,
        help='Length of state vectors generated (default: 50)')

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
