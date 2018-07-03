import os
import numpy as np
import tensorflow as tf

sess = tf.Session()

## Define the network graph
x = tf.placeholder(dtype=tf.float32, shape=[None, 200])
y = tf.placeholder(dtype=tf.int32, shape=[None])

hidden1 = tf.layers.dense(x, 100, activation=tf.nn.relu, name="hidden1")
hidden2 = tf.layers.dense(hidden1, 50, activation=tf.nn.relu, name="hidden2")
hidden3 = tf.layers.dense(hidden2, 25, activation=tf.nn.relu, name="hidden3")

logits = tf.layers.dense(hidden3, 4, activation=tf.nn.relu, name="logits")

prediction = tf.argmax(logits, 1, name="prediction")
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
    logits=logits, name="loss"))

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess.run(tf.global_variables_initializer())

## Helper functions for handling data
def load_data(dir, name):
    """Loads a .npz file from the data directory"""
    if name.split('.')[-1] != 'npz':
        name += '.npz'
    f = os.path.join(dir, name)
    # Load the data from the file
    d = np.load(f)
    states = d['states']
    labels = d['labels']
    # Format the data for the network
    states = np.abs(states)
    states[np.isnan(states)] = 0.0
    return states, labels

def fetch_batch(data, labels, n, b_size):
    """Returns the nth batch of size b_size from data and labels"""
    batch_data = data[n*b_size:(n+1)*b_size]
    batch_labels = labels[n*b_size:(n+1)*b_size]
    return batch_data, batch_labels

cur_dir = os.path.abspath(os.path.join(__file__, os.pardir))
data_dir = os.path.join(cur_dir, "data")

## Train the network
def train():
    train_states, train_labels = load_data(data_dir, "train")
    for epoch in range(500):
        if epoch % 50 == 0:
            print("Epoch: {}".format(epoch))
        for b_n in range(5):
            batch = fetch_batch(train_states, train_labels, b_n, 1000)
            sess.run(train_op, feed_dict={x: batch[0], y: batch[1]})
    print("Training completed.")

## Test the network
def test():
    test_states, test_labels = load_data(data_dir, "test")
    test_predictions = sess.run(prediction, feed_dict={ x: test_states })
    n_correct = np.sum(test_predictions == test_labels)
    accuracy = float(n_correct)/len(test_labels)
    conf_mat = sess.run(tf.confusion_matrix(test_labels, test_predictions))
    print("Network classifed {}/{} states correctly ({}%)".format(n_correct, len(test_labels), accuracy*100))
    print("Confusion matrix:")
    print(conf_mat)

def classify(data):
    """Classifies data using the trained network"""
    pred = sess.run(prediction, feed_dict={x: [data]})
    return pred[0]

if __name__ == '__main__':
    train()
    test()
