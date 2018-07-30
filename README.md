# Quantum Optics   

Python code with the ultimate goal of classifying quantum states of light using
a neural network.

The `quoptics` module contains helper functions for creating some more exotic
states, i.e. cat states, using [QuTiP](http://qutip.org). It also contains
the `NeuralNetwork` class for classifying states.   
The [legacy](legacy/quoptics) folder contains code that was written before I
discovered QuTiP, that is all already implemented in QuTiP.

To ensure the required packages are installed, run
`pip3 install -r requirements.txt`   

Running `python3 data/gen_data.py` will generate training/test/validation data
for the network in the `data` folder. The data is produced as an array of states
 and an array of labels, such that `states[n]` is labelled by `label[n]`. The
 labels are integers labelling which type the state is according to:

| Label  |  State Type  |
| ------ | ------------ |
|   0    |     Fock     |
|   1    |     Cat      |
|   2    | Squeezed Cat |
|   3    |    Zombie    |
|   4    |  Cubic Phase |
|   5    |      ON      |

Running `python3 test_network.py` will load the pre-trained model from the
`model` folder, and test it's accuracy against the data in `data/test.npz`.

The file [cnn.py](cnn.py) is the start of a convolutional neural network that
classifies states by analysing a plot of their Wigner function.
