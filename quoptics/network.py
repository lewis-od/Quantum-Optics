import functools
import numpy as np
import tensorflow as tf

class NeuralNetwork(object):
    r"""
    A tensorflow neural network for classifying quantum states of light

    When inputting a state

    .. math::

        \lvert \psi \rangle = \sum_{n=o}^{\infty} c_n \lvert n \rangle

    where the :math:`\lvert n \rangle` are Fock states, the network expects the
    state data to be in the form of a numpy array, where the :math:`n` th
    element is the modulus of :math:`c_n`.

    :param model_dir: The directory to save/load the model data to/from
    :param dropout: The dropout probability to use when training
    :param learning_rate: The learning rate to use when training
    """
    def __init__(self, model_dir=None, dropout=0.2, learning_rate=0.001):
        self.__learning_rate = learning_rate
        self.__create_estimator = functools.partial(self.__estimator_factory, model_dir, dropout)
        self.estimator = self.__create_estimator(learning_rate)

    def __estimator_factory(self, model_dir, dropout, learning_rate):
        coefficients = tf.feature_column.numeric_column('coefficients',
            shape=[25])
        estimator = tf.estimator.DNNClassifier(
                feature_columns=[coefficients],
                hidden_units=[25, 25, 10],
                n_classes=6,
                optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                dropout=dropout,
                model_dir=model_dir
        )
        return estimator

    def __input_fn(self, fname):
        data = np.load(fname)
        states = np.abs(data['states'])
        labels = data['labels'].astype(int)

        return { 'coefficients': states }, labels

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, val):
        self.__learning_rate = val
        self.estimator = self.__create_estimator(val)

    def train(self, train_file, steps):
        r"""
        Train the neural network

        :param train_file: The file containing the training data
        :param steps: How many training steps to run
        """
        train_fn = functools.partial(self.__input_fn, train_file)
        self.estimator.train(input_fn=train_fn, steps=steps)

    def test(self, test_file):
        r"""
        Test the neural network against a test dataset

        :param test_file: The file containing the test data
        :returns metrics: A dictionary containing various metrics about the
            accuracy of the network
        :returns conf_mat: The confusion matrix. Rows represent actual labels
            and columns represent predicted labels
        """
        test_fn = functools.partial(self.__input_fn, test_file)
        metrics = self.estimator.evaluate(input_fn=test_fn, steps=1)

        test_states, actual_labels = test_fn()
        test_states = test_states['coefficients']
        predictions = [p['class_ids'][0] for p in self.predict(test_states)]
        with tf.Session().as_default():
            conf_mat = tf.confusion_matrix(actual_labels, predictions).eval()
        return metrics, conf_mat

    def predict(self, states):
        r"""
        Run the input array through the
        `tensorflow.estimator.Estimator.predict <https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#predict>`_
        method

        :param states: A numpy array containing multiple test_states
        :returns: A dictionary containing the classification information
        """
        pred_input = tf.estimator.inputs.numpy_input_fn(
            { 'coefficients': states }, batch_size=1, shuffle=False)
        prediction = list(self.estimator.predict(pred_input))
        return prediction

    def classify(self, state):
        r"""
        Classify a state using the network

        :param state: The state to classify
        :returns: An integer indicating the category the state belongs to
        """
        prediction = self.predict(np.array([state]))[0]
        return prediction['class_ids'][0]

    def classify_dist(self, state):
        r"""
        Classify a state using the network

        :param state: The state to classify
        :returns: An array containing a probability distribution, where the nth
            entry is the probability that the state belongs to category n, as
            predicted by the network
        """
        prediction = self.predict(np.array([state]))[0]
        return prediction['probabilities']
