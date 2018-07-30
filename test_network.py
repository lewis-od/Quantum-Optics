import os
import tensorflow as tf
from quoptics.network import NeuralNetwork

tf.logging.set_verbosity(tf.logging.WARN)

cur_dir = os.path.abspath('.')
model_dir=os.path.join(cur_dir, "model")

net = NeuralNetwork(model_dir=model_dir)

metrics, conf_mat = net.test(os.path.join(cur_dir, "data", "test.npz"))
print("Metrics:")
print(metrics)
print("Confusion Matrix:")
conf_mat_str = "    " + str(conf_mat).replace("\n", "\n    ")
print(conf_mat_str)
