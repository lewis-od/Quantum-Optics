import os
import functools
import numpy as np
import tensorflow as tf

class NeuralNetwork(object):
    def __init__(self, model_dir=None, dropout=0.2, learning_rate=0.001):
        self.estimator = self.__create_estimator(model_dir, dropout, learning_rate)

    def __create_estimator(self, model_dir, dropout, learning_rate):
        coefficients = tf.feature_column.numeric_column('coefficients',
            shape=[25])
        estimator = tf.estimator.DNNClassifier(
                feature_columns=[coefficients],
                hidden_units=[25, 25, 10],
                n_classes=4,
                optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                dropout=dropout,
                model_dir=model_dir
        )
        return estimator

    def __input_fn(self, fname):
        data = np.load(fname)
        states = np.abs(data['states'])
        states = np.array([state[0:25] for state in states])
        labels = data['labels'].astype(int)

        return { 'coefficients': states }, labels

    def train(self, train_file, steps):
        train_fn = functools.partial(self.__input_fn, train_file)
        self.estimator.train(input_fn=train_fn, steps=steps)

    def test(self, test_file):
        test_fn = functools.partial(self.__input_fn, test_file)
        metrics = self.estimator.evaluate(input_fn=test_fn, steps=1)

        test_states, actual_labels = test_fn()
        test_states = test_states['coefficients']
        predictions = [p['class_ids'][0] for p in self.predict(test_states)]
        with tf.Session().as_default():
            conf_mat = tf.confusion_matrix(actual_labels, predictions).eval()
        return metrics, conf_mat

    def predict(self, states):
        pred_input = tf.estimator.inputs.numpy_input_fn(
            { 'coefficients': states}, batch_size=1, shuffle=False)
        prediction = list(self.estimator.predict(pred_input))
        return prediction

    def classify(self, state):
        prediction = self.predict(np.array([state]))[0]
        return prediction['class_ids'][0]

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    cur_dir = os.path.abspath('.')
    model_dir=os.path.join(cur_dir, "model")

    net = NeuralNetwork(model_dir=model_dir)

    metrics, conf_mat = net.test(os.path.join(cur_dir, "data", "test.npz"))
    print("Metrics:")
    print(metrics)
    print("Confusion Matrix:")
    conf_mat_str = "    " + str(conf_mat).replace("\n", "\n    ")
    print(conf_mat_str)

# test_states, actual = input_fn_test()
# # Workaround for tensorflow bug
# # (see https://github.com/tensorflow/tensorflow/issues/11621)
# conf_input = tf.estimator.inputs.numpy_input_fn(
#     test_states, batch_size=1, shuffle=False)
# predictions = list(estimator.predict(conf_input))
#
# predictions = [d['class_ids'][0] for d in predictions]
# with tf.Session().as_default():
#     conf_mat = tf.confusion_matrix(actual, predictions).eval()
# print(conf_mat)
