import tensorflow as tf

## Define the network
x = tf.placeholder(dtype=tf.float32, shape=[None, 100])
y = tf.placeholder(dtype=tf.int32, shape=[None])

layer1 = tf.contrib.layers.fully_connected(x, 50, tf.nn.relu)
layer2 = tf.contrib.layers.fully_connected(layer1, 25, tf.nn.relu)
logits = tf.contrib.layers.fully_connected(layer2, 3, tf.nn.sigmoid)

# Use softmax loss function (mutually exclusive categories)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

# Use ADAM algorithm for optimisation
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


## Run the network
sess = tf.Session()
sess.run(tf.global_variables_initializer())
