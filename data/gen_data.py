import sys
import os
import argparse
import numpy as np
# TODO: There must be a better way to do this?
directory = os.path.abspath(os.path.join(__file__, os.pardir))
parent = os.path.abspath(os.path.join(directory, os.pardir))
sys.path.append(parent)

from quoptics.states import random_states

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
    parser.add_argument('--truncation', type=int, required=False, default=80,
        help='Size of matrices to use when calculating states (default: 100)',
        metavar='T')
    parser.add_argument('--cutoff', type=int, required=False, default=25,
        help='Length of state vectors generated (default: 25)', metavar='LEN')

    # Parse arguments
    params = parser.parse_args()

    # Set truncation and cutoff
    T = params.truncation
    C = params.cutoff

    print("Generating data with params {}".format(params.__dict__))

    # Generate data
    training = random_states(T, params.training, cutoff=C, qutip=False)
    print("Generated training data")
    test = random_states(T, params.test, cutoff=C, qutip=False)
    print("Generated test data")
    validation = random_states(T, params.validation, cutoff=C, qutip=False)
    print("Generated validation data")

    # Save data
    save_data(training, "train")
    save_data(test, "test")
    save_data(validation, "validation")
