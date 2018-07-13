import os
import glob
import imageio
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 200, 200, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5],
        padding="same", activation=tf.nn.relu)

    # Output of pool1 has shape [batch_size, 100, 100, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
        padding="same", activation=tf.nn.relu)

    # Output of pool2 has shape [batch_size, 50, 50, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[10, 10],
        padding="same", activation=tf.nn.relu)

    # Output of pool 3 has shape [batch_size, 10, 10, 64]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[5, 5], strides=5)

    pool3_flat = tf.reshape(pool3, [-1, 10*10*64])
    dense = tf.layers.dense(pool3_flat, 1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4,
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(dropout, 4)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # EVAL mode
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
            predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(args):
    pass

if __name__ == '__main__':

    cur_dir = os.path.abspath(".")
    image_dir = os.path.join(cur_dir, "image_data", "training")
    # Load training images
    image_path = os.path.join(image_dir, "wigner_*.png")
    file_names = glob.glob(image_path)
    images = np.empty([len(file_names), 200, 200])
    for n, fname in enumerate(file_names):
        imdata = imageio.imread(fname, as_gray=True)
        imdate = imdata.reshape(200*200)
        images[n] = imdata
    # Load training labels
    labels = np.load(os.path.join(image_dir, 'labels.npy'))
    labels = labels.astype(dtype=int)
    state_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=os.path.join(cur_dir, "cnn_summary"))

    tensors_to_log = { "probabilities": "softmax_tensor" }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input = tf.estimator.inputs.numpy_input_fn(
        x={ "x": images },
        y=labels,
        batch_size = 1,
        num_epochs=None,
        shuffle=True
    )

    state_classifier.train(input_fn=train_input, steps=500, hooks=[logging_hook])

    tf.app.run()
