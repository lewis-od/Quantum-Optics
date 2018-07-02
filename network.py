import numpy as np
import tensorflow as tf
import os
import sys

# Number of categories to classify states into
N_TYPES = 4

## Define the network
x = tf.placeholder(dtype=tf.float32, shape=[None, 200])
y = tf.placeholder(dtype=tf.int32, shape=[None])

layer1 = tf.contrib.layers.fully_connected(x, 100, tf.nn.relu)
layer2 = tf.contrib.layers.fully_connected(layer1, 50, tf.nn.relu)
layer3 = tf.contrib.layers.fully_connected(layer2, 25, tf.nn.relu)
logits = tf.contrib.layers.fully_connected(layer3, N_TYPES, tf.nn.sigmoid)

# Use softmax loss function (mutually exclusive categories)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Use ADAM algorithm for optimisation
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

## Load the training data
def load_data(data_dir, fname):
    train_path = os.path.join(data_dir, fname)
    train_file = np.load(train_path)

    states = train_file['states']
    labels = train_file['labels']

    # Use |c_n| as training input, since physics depends on |c_n|^2
    states = np.abs(states)
    states[np.isnan(states)] = 0

    return states, labels

# Get a batch from the training data
def fetch_batch(batch_index, batch_size):
    batch_states = states[batch_index*batch_size:(batch_index+1*batch_size)]
    batch_labels = labels[batch_index*batch_size:(batch_index+1*batch_size)]
    return batch_states, batch_labels

# Shuffle states and labels
def shuffle_data(states, labels):
    new_states = np.empty(states.shape, dtype=states.dtype)
    new_labels = np.empty(labels.shape, dtype=labels.dtype)
    perm = np.random.permutation(len(labels))
    for old, new in enumerate(perm):
        new_states[new] = states[old]
        new_labels[new] = labels[old]
    return new_states, new_labels

# Train the network
def train(states, labels):
    for epoch in range(501):
        for b_n in range(5):
            batch = fetch_batch(b_n, 1000)
            _, accuracy_val = sess.run([train_op, accuracy],
                feed_dict={ x: batch[0], y: batch[1] })
        # states, labels = shuffle_data(states, labels)
        if epoch % 10 == 0:
            print("Epoch: {}".format(epoch))

# Test the network against the test data
def test(data_dir):
    test_states, test_labels = load_data(data_dir, "test.npz")

    predicted = sess.run([correct_pred], feed_dict={x: test_states})[0]
    predicted = np.array(predicted)

    loss_val = sess.run(loss, feed_dict={x: test_states, y: test_labels})
    print("Loss value is: {}".format(loss_val))

    n_correct = np.sum(predicted == test_labels)
    test_accuracy = float(n_correct) / float(len(test_labels))

    conf_mat = sess.run(tf.confusion_matrix(test_labels, predicted))

    print("{} states out of {} correctly classified".format(n_correct, len(test_labels)))
    print("Test accuracy is: {}".format(test_accuracy))
    print("Confusion matrix:")
    print(conf_mat)

def classify(state):
    data = np.abs(state.data)
    data[np.isnan(data)] = 0.0
    types = ['fock', 'coherent', 'squeezed', 'cat']
    index = sess.run(correct_pred, feed_dict={x: [data]})[0]
    return types[index]

# Initialise the session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
cur_dir = os.path.abspath(os.path.join(__file__, os.pardir))
data_dir = os.path.join(cur_dir, "data")

## Train network
# states, labels = load_data(data_dir, "train.npz")
# train(states, labels)

## Use trained network
saver = tf.train.Saver()
saver.restore(sess, os.path.join(cur_dir, "weights", "model.ckpt"))

test(data_dir)
