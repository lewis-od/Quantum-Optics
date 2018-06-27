import sys
import os
# TODO: There must be a better way to do this?
parent = os.path.abspath(os.path.join(__file__, os.pardir))
parent = os.path.abspath(os.path.join(parent, os.pardir))
sys.path.append(parent)

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
        state = np.zeros([qo.conf.T + 1])
        # Generate the state
        if type == 'fock':
            param = np.random.randint(0, qo.conf.T)
            state[:-1] = qo.states.fock(param)
        elif type == 'coherent':
            state[:-1] = qo.states.coherent(2)
        elif type == 'squeezed':
            state[:-1] = qo.states.squeezed(2)
        # Last entry of array is a number identifying the type of state
        state[-1] = id
        states[i] = state
    return states

def parse_arg(n):
    argument = None
    try:
        argument = int(sys.argv[n])
    except:
        pass
    return argument

if __name__ == '__main__':
    # Default values - training, test, validation, truncation
    params = [10, 5, 5, qo.conf.T]

    # Name of file is first argument
    n_args = len(sys.argv) - 1
    if n_args > 4:
        print("Too many arguments supplied - maximum is 3")
        sys.exit()

    # Parse all given arguments
    for n in range(n_args):
        params[n] = parse_arg(n+1)

    if np.any([p is None for p in params]):
        print("All arguments must be integers")
        sys.exit()

    # Set module-wide truncation
    qo.conf.T = params[3]

    print("Generating data with (train,test,val,T) = {}".format(params))
    training = gen_states(params[0])
    print(training)
