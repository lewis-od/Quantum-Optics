clear all;
% Load Python
[ver, exec, loaded] = pyversion;
if (loaded == 0)
    pyversion '/usr/local/bin/python3'
end

% Create a NeuralNetwork object (from network.py)
net = py.quoptics.network.NeuralNetwork("model");

% Run the network against the test data and print the results
net.test("data/test.npz")
