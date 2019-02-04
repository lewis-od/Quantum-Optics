# Quantum Optics   

A neural network to classify quantum states of light.

The `quoptics` module contains the `NeuralNetwork` class for classifying states.
It also contains helper functions for creating some more exotic
states, i.e. cat states, using [QuTiP](http://qutip.org). 

To ensure the required packages are installed, run
`pip3 install -r requirements.txt`   

Running `python3 test_network.py` will load the pre-trained model from the
`model` folder, and test it's accuracy against the data in `data/test.npz`.

The file `interop.m` gives an example of how to call the neural network
from MATLAB.

A whitepaper describing the usage and training of the neural network in more
detail, as well as it's integration into 
[AdaQuantum](https://github.com/paulk444/AdaQuantum) can be found at 
[arXiv:1812.03183](https://arxiv.org/abs/1812.03183)

Full documentation for the quoptics module is available to view
[here](https://lewis-od.github.io/Quantum-Optics/).
