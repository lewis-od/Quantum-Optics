clear all;
% Load Python
[ver, exec, loaded] = pyversion;
if (loaded == 0)
    pyversion '/usr/local/bin/python3'
end

% Create a tensorflow session
sess = py.tensorflow.Session();
% Create a NeuralNetwork object (from network.py)
net = py.network.NeuralNetwork(sess);
% Load the pre-trained network
net.restore("weights");

% Run the network agains the test data and print the results
net.test()

% Clean up
py.tensorflow.reset_default_graph();
sess.close()