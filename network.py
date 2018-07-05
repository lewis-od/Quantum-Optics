import os
import shutil
import argparse
import json
import numpy as np
import tensorflow as tf

## Hyperparameters

LEARNING_RATE = 0.001
KEEP_PROB = 0.3

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and test the dnn")
    parser.add_argument('--learning-rate',
        type=float, required=False,
        metavar='RATE', default=0.001,
        help="Learning rate to used in optimization")
    parser.add_argument('--keep-prob',
        type=float, required=False,
        metavar='PROB', default=0.3,
        help="Keep probability to used in dropout training")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--restore',
        required=False, type=str, metavar='DIR',
        help='Load the saved parameter values from DIR instead of training the network')
    group.add_argument('--save',
        required=False, type=str, metavar='DIR',
        help='Save the parameter values to DIR after training')

    params = parser.parse_args()

    # Set values of hyperparameters to those supplied on the command line
    KEEP_PROB = params.keep_prob
    LEARNING_RATE = params.learning_rate

## Helper functions for creating network

def weight_variable(shape):
    """Creates a weight variable with appropriate initialisation"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Creates a bias variable with appropriate initialisation"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
  """Attach summaries to a tensor for visualisation in tensorboard"""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, name, act=tf.nn.relu):
    """Create a neural network layer with summaries attached"""
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activations')
        tf.summary.histogram('activations', activations)
        return activations

# Create a tensorflow session
sess = tf.Session()

## Define the network graph

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='x_input')
    y_ = tf.placeholder(dtype=tf.int64, shape=[None], name='y_input')

hidden1 = nn_layer(x, 100, 50, 'layer1')

with tf.name_scope("droput"):
    keep_prob = tf.placeholder(dtype=tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

hidden2 = nn_layer(dropped, 50, 25, 'layer2')
# Don't apply softmax activation yet, it's applied automatically when
# calculating the loss
y = nn_layer(hidden2, 25, 4, 'layer3', act=tf.identity)

# Use cross entropy loss function
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
tf.summary.scalar('cross_entropy', cross_entropy)

# Train using the ADAM optimiser
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE).minimize(cross_entropy)

# Calculate the accuracy of the network
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        prediction = tf.argmax(y, 1)
        correct_pred = tf.equal(prediction, y_)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all summary nodes into one
merged_summs = tf.summary.merge_all()
# Initialise all variables
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
    """Trains the network and tests its accuracy"""
    # Load training and test data
    train_states, train_labels = load_data(data_dir, "train")
    test_states, test_labels = load_data(data_dir, "test")
    # Directory to save variable summaries to
    summary_dir = os.path.join(cur_dir, "summary")
    if os.path.isdir(summary_dir): shutil.rmtree(summary_dir)
    # These write the network data for visualization later in tensorboard
    train_writer = tf.summary.FileWriter(
        os.path.join(summary_dir, "train"), sess.graph)
    test_writer = tf.summary.FileWriter(
        os.path.join(summary_dir, "test"))

    for epoch in range(500):
        if epoch % 10 == 0:
            # Evaluate network performance every 10 epochs
            summary, acc = sess.run([merged_summs, accuracy],
                feed_dict={ x: test_states,
                            y_: test_labels,
                            keep_prob: 1.0
            })
            test_writer.add_summary(summary, epoch)
            print("Accuracy at step {} is : {}".format(epoch, acc))
        else:
            # TODO: More elegant batch training
            for b_n in range(5):
                batch = fetch_batch(train_states, train_labels, b_n, 1000)
                if epoch % 100 == 99 and b_n == 1:
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged_summs, train_op],
                        feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB},
                        run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d-%01d' % (epoch, b_n))
                    train_writer.add_summary(summary, epoch)
                else:
                    summary, _ = sess.run([merged_summs, train_op],
                        feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB})
                    train_writer.add_summary(summary, epoch)
    train_writer.close()
    test_writer.close()
    print("Training completed.")
    print('-'*80)

## Test the network
def test():
    """Tests the network and prints out various statistics"""
    # Load test data
    test_states, test_labels = load_data(data_dir, "test")
    # Test the network
    test_predictions, acc = sess.run([prediction, accuracy],
        feed_dict={ x: test_states, y_: test_labels, keep_prob: 1.0 })
    n_correct = np.sum(test_predictions == test_labels)
    conf_mat = sess.run(tf.confusion_matrix(test_labels, test_predictions))
    # Print the accuracy and confusion matrix
    print("Network classifed {}/{} states correctly ({:.2f}%)".format(n_correct,
        len(test_labels), acc*100))
    print("Hyperparameters:")
    print("    Learning rate: {}".format(LEARNING_RATE))
    print("    Keep probability: {}".format(KEEP_PROB))
    print("Confusion matrix:")
    print(conf_mat)

def classify(data):
    """Classifies data using the trained network"""
    pred = sess.run(prediction, feed_dict={x: [data]})
    return pred[0]

def save(model_dir):
    """Saves the values of weights and biases, as well as the hyperparameters"""
    # Save model data
    saver = tf.train.Saver()
    model_file = os.path.join(cur_dir, model_dir, "model.ckpt")
    saver.save(sess, model_file)
    # Save hyperparameters
    hyperparams_file = os.path.join(cur_dir, model_dir, "hyperparameters.json")
    hyperparams = { 'keep_prob': KEEP_PROB, 'learning_rate': LEARNING_RATE }
    with open(hyperparams_file, 'w') as f:
        json.dump(hyperparams, f)

def restore(model_dir):
    """Loads the saved model and hyperparameters from the weights folder"""
    # Load model data
    loader = tf.train.Saver()
    model_file = os.path.join(cur_dir, model_dir, "model.ckpt")
    loader.restore(sess, model_file)
    # Load hyperparameters
    hyperparams_file = os.path.join(cur_dir, model_dir, "hyperparameters.json")
    with open(hyperparams_file, 'r') as f:
        params = json.load(f)
    global KEEP_PROB
    KEEP_PROB = params['keep_prob']
    global LEARNING_RATE
    LEARNING_RATE = params['learning_rate']

if __name__ == '__main__':
    # Restore or train the network depending on command line input
    if params.restore is not None:
        restore(params.restore)
    else:
        train()
    # Test the network
    test()

    # Save the parameters of the trained network if requested by user
    if params.save is not None:
        save(params.save)
