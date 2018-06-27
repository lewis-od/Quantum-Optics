# Quantum Optics   

Python code with the ultimate goal of classifying quantum states of light using
a neural network

The `quoptics` module contains useful code for producing various states (e.g.
coherent states, squeezed states, etc). It also has matrix representations of various
operators.
The vectors it produces are in the Fock basis, so the nth component of the
vector is the coefficient of |n>, or equivalently `psi[n] = <psi|n>`

Run `python data/gen_data.py` to generate training/test/validation data for the network. The data is produced as an array of states such that `state[0:-1]` contains the actual state data, and `state[-1]` is an integer labelling what type the state is accoring to:

| Label  | State Type |
| ------ | ---------- |
|    0   |    Fock    |
|    1   |  Coherent  |
|    2   |  Squeezed  |
