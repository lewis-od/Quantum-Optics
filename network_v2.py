import os
import shutil
import argparse
import json
import numpy as np
import tensorflow as tf


class NeuralNetwork(object):

    def __init__(self, sess, learning_rate=0.001, keep_prob=0.3,
        summary_dir="summary", data_dir="data", train_file="train.npz",
        test_file="test.npz"):
        self.sess = sess # tensorflow session
        # Network hyperparameters
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        # Current directory
        cur_dir = os.path.abspath(os.path.join(__file__, os.pardir))
        # Directory to output tensorboard summary data to
        if summary_dir == os.path.abspath(summary_dir):
            self.summary_dir = summary_dir
        else:
            self.summary_dir = os.path.join(cur_dir, summary_dir)
        # Directory containing training/test data
        if data_dir == os.path.abspath(data_dir):
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(cur_dir, data_dir)
        self.train_file = train_file
        self.test_file = test_file

        # Operations
        self.x_input = None
        self.y_input = None
        self.keep_prob_input = None
        self.cross_entropy = None
        self.accuracy = None
        self.train_op = None
        self.prediction = None
        self.correct_pred = None
        self.summaries = None

        self._create_graph()
        self.sess.run(tf.global_variables_initializer())

    def _weight_variable(self, shape):
        """Creates a weight variable with appropriate initialisation"""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        """Creates a bias variable with appropriate initialisation"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _variable_summaries(self, var):
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

    def _nn_layer(self, input_tensor, input_dim, output_dim,
        name, act=tf.nn.relu):
        """Create a neural network layer with summaries attached"""
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                weights = self._weight_variable([input_dim, output_dim])
                self._variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self._bias_variable([output_dim])
                self._variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activations')
            tf.summary.histogram('activations', activations)
            return activations

    def _create_graph(self):
        """Creates the neural network computaion graph"""
        with tf.name_scope('input'):
            self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, 100],
                name='x_input')
            self.y_input = tf.placeholder(dtype=tf.int64, shape=[None],
                name='y_input')

        hidden1 = self._nn_layer(self.x_input, 100, 50, 'layer1')

        with tf.name_scope("droput"):
            self.keep_prob_input = tf.placeholder(dtype=tf.float32)
            tf.summary.scalar('dropout_keep_probability', self.keep_prob_input)
            dropped = tf.nn.dropout(hidden1, self.keep_prob_input)

        hidden2 = self._nn_layer(dropped, 50, 25, 'layer2')
        # Don't apply softmax activation yet, it's applied automatically when
        # calculating the loss
        y = self._nn_layer(hidden2, 25, 4, 'layer3', act=tf.identity)

        # Use cross entropy loss function
        with tf.name_scope('cross_entropy'):
            self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                labels=self.y_input, logits=y)
        tf.summary.scalar('cross_entropy', self.cross_entropy)

        # Train using the ADAM optimiser
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.cross_entropy)

        # Calculate the accuracy of the network
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                self.prediction = tf.argmax(y, 1)
                self.correct_pred = tf.equal(self.prediction, self.y_input)
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(
                    tf.cast(self.correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()

    def load_data(self, name):
        """Loads a .npz file from the data directory"""
        if name.split('.')[-1] != 'npz':
            name += '.npz'
        f = os.path.join(self.data_dir, name)
        # Load the data from the file
        d = np.load(f)
        states = d['states']
        labels = d['labels']
        # Format the data for the network
        states = np.abs(states)
        states[np.isnan(states)] = 0.0
        return states, labels

    def fetch_batch(self, data, labels, n, b_size):
        """Returns the nth batch of size b_size from data and labels"""
        batch_data = data[n*b_size:(n+1)*b_size]
        batch_labels = labels[n*b_size:(n+1)*b_size]
        return batch_data, batch_labels

    def train(self, n_epochs):
        """Trains the network and tests its accuracy"""
        # Load training and test data
        train_states, train_labels = self.load_data(self.train_file)
        test_states, test_labels = self.load_data(self.test_file)
        # Directory to save variable summaries to
        if os.path.isdir(self.summary_dir): shutil.rmtree(self.summary_dir)
        # These write the network data for visualization later in tensorboard
        train_writer = tf.summary.FileWriter(
            os.path.join(self.summary_dir, "train"), self.sess.graph)
        test_writer = tf.summary.FileWriter(
            os.path.join(self.summary_dir, "test"))

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                # Evaluate network performance every 10 epochs
                summary, acc = self.sess.run([self.summaries, self.accuracy],
                    feed_dict={ self.x_input: test_states,
                                self.y_input: test_labels,
                                self.keep_prob_input: 1.0
                })
                test_writer.add_summary(summary, epoch)
                print("Accuracy at step {} is : {}".format(epoch, acc))
            else:
                # TODO: More elegant batch training
                for b_n in range(5):
                    batch = self.fetch_batch(
                        train_states, train_labels, b_n, 1000)
                    if epoch % 100 == 99 and b_n == 1:
                        run_metadata = tf.RunMetadata()
                        summary, _ = self.sess.run(
                            [self.summaries, self.train_op],
                            feed_dict={
                                self.x_input: batch[0],
                                self.y_input: batch[1],
                                self.keep_prob_input: self.keep_prob
                            },
                            run_metadata=run_metadata)
                        train_writer.add_run_metadata(
                            run_metadata, 'step%03d-%01d' % (epoch, b_n))
                        train_writer.add_summary(summary, epoch)
                    else:
                        summary, _ = self.sess.run(
                            [self.summaries, self.train_op],
                            feed_dict={
                                self.x_input: batch[0],
                                self.y_input: batch[1],
                                self.keep_prob_input: self.keep_prob
                            })
                        train_writer.add_summary(summary, epoch)
        train_writer.close()
        test_writer.close()
        print("Training completed.")
        print('-'*80)

    def test(self):
        """Tests the network and prints out various statistics"""
        # Load test data
        test_states, test_labels = self.load_data(self.test_file)
        # Test the network
        test_predictions, acc = self.sess.run([self.prediction, self.accuracy],
            feed_dict={
                self.x_input: test_states,
                self.y_input: test_labels,
                self.keep_prob_input: 1.0
            })
        n_correct = np.sum(test_predictions == test_labels)
        conf_mat = self.sess.run(
            tf.confusion_matrix(test_labels, test_predictions))
        # Print the accuracy and confusion matrix
        print("Network classifed {}/{} states correctly ({:.2f}%)".format(
            n_correct, len(test_labels), acc*100))
        print("Hyperparameters:")
        print("    Learning rate: {}".format(self.learning_rate))
        print("    Keep probability: {}".format(self.keep_prob))
        print("Confusion matrix:")
        print(conf_mat)

    def classify(self, data):
        """Classifies data using the trained network"""
        pred = self.sess.run(self.prediction,
            feed_dict={
                self.x_input: [data],
                self.keep_prob_input: 1.0
            })
        return pred[0]

    def save(self, model_dir):
        """Saves the values of weights and biases, as well as the hyperparameters"""
        # Save model data
        saver = tf.train.Saver()
        cur_dir = os.path.abspath(os.path.join(__file__, os.pardir))
        model_file = os.path.join(cur_dir, model_dir, "model.ckpt")
        saver.save(self.sess, model_file)
        # Save hyperparameters
        hyperparams_file = os.path.join(cur_dir, model_dir, "hyperparameters.json")
        hyperparams = {
            'keep_prob': self.keep_prob,
            'learning_rate': self.learning_rate,
        }
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f)
