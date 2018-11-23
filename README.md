# Quantum Optics   

A neural network to classify quantum states of light.

The `quoptics` module contains the `NeuralNetwork` class for classifying states.  
It also contains helper functions for creating some more exotic
states, i.e. cat states, using [QuTiP](http://qutip.org). 

To ensure the required packages are installed, run
`pip3 install -r requirements.txt`   

Running `python3 test_network.py` will load the pre-trained model from the
`model` folder, and test it's accuracy against the data in `data/test.npz`.

The full documentation for the quoptics module is available to view
[here](https://lewis-od.github.io/Quantum-Optics/).
