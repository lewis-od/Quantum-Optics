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

Running `python3 test_network.py` will load the pre-trained model from the
`model` folder, and test it's accuracy against the data in `data/test.npz`.

The file [cnn.py](cnn.py) is the start of a convolutional neural network that
classifies states by analysing a plot of their Wigner function.   

The full documentation for the quoptics module is available to view
[here](https://lewis-od.github.io/Quantum-Optics/).
