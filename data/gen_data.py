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
    states = np.zeros([n, qo.conf.T + 1])
    for i in range(n):
        # Cycle through the types of state
        id = i % len(types) # Integer identifying type of state
        type = types[id]
        # TODO: Made state a 2T dimensional vector to store real/complex or
        # modulus/argument as separate features
        state = np.zeros([qo.conf.T + 1])
        # Generate the state
        if type == 'fock':
            param = np.random.randint(0, qo.conf.T)
            state[:-1] = qo.states.fock(param)
        elif type == 'coherent':
            # TODO: Generate random complex number
            state[:-1] = qo.states.coherent(2)
        elif type == 'squeezed':
            # TODO: Generate random complex number
            state[:-1] = qo.states.squeezed(2)
        # Last entry of array is a number identifying the type of state
        state[-1] = id
        states[i] = state
    return states

def save_data(data, name):
    output = os.path.join(directory, name + ".npy")
    if os.path.isfile(output):
        print("File {} already exists, do you want to overwrite it? (y/n)".format(output))
        res = input()
        while res not in ['y', 'n']:
            print("Invalid input: " + res)
            res = input()
        if res == 'n':
            return
    try:
        np.save(output, data)
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
