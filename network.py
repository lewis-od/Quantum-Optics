import os
import numpy as np
import tensorflow as tf

## Helper functions for creating network

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
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

sess = tf.Session()

## Define the network graph
with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 200], name='x_input')
    y_ = tf.placeholder(dtype=tf.int64, shape=[None], name='y_input')

hidden1 = nn_layer(x, 200, 50, 'layer1')
# Don't apply softmax activation yet, it's applied automatically when
# calculating the loss
y = nn_layer(hidden1, 50, 4, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        prediction = tf.argmax(y, 1)
        correct_pred = tf.equal(prediction, y_)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged_summs = tf.summary.merge_all()
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
    test_states, test_labels = load_data(data_dir, "test")
    train_writer = tf.summary.FileWriter(os.path.join(cur_dir, "summary", "train"), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(cur_dir, "summary", "test"))

    for epoch in range(500):
        if epoch % 10 == 0:
            summary, acc = sess.run([merged_summs, accuracy], feed_dict={x: test_states, y_: test_labels})
            test_writer.add_summary(summary, epoch)
            print("Accuracy at step {} is : {}".format(epoch, acc))
        else:
            for b_n in range(5):
                batch = fetch_batch(train_states, train_labels, b_n, 1000)
                if epoch % 100 == 99 and b_n == 1:
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged_summs, train_op],
                        feed_dict={x: batch[0], y_: batch[1]},
                        run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d-%01d' % (epoch, b_n))
                    train_writer.add_summary(summary, epoch)
                else:
                    summary, _ = sess.run([merged_summs, train_op],
                        feed_dict={x: batch[0], y_: batch[1]})
                    train_writer.add_summary(summary, epoch)
    train_writer.close()
    test_writer.close()
    print("Training completed.")

## Test the network
def test():
    test_states, test_labels = load_data(data_dir, "test")
    test_predictions, acc = sess.run([prediction, accuracy],
        feed_dict={ x: test_states, y_: test_labels })
    n_correct = np.sum(test_predictions == test_labels)
    conf_mat = sess.run(tf.confusion_matrix(test_labels, test_predictions))
    print("Network classifed {}/{} states correctly ({:.2f}%)".format(n_correct,
        len(test_labels), acc*100))
    print("Confusion matrix:")
    print(conf_mat)

def classify(data):
    """Classifies data using the trained network"""
    pred = sess.run(prediction, feed_dict={x: [data]})
    return pred[0]

def restore():
    """Loads the saved model from the weights folder"""
    loader = tf.train.Saver()
    model_file = os.path.join(cur_dir, "weights", "model.ckpt")
    loader.restore(sess, model_file)

if __name__ == '__main__':
    train()
    test()
