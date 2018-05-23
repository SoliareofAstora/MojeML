from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Training Parameters
learning_rate = 0.01


display_step = 1000
examples_to_show = 10

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 784])
num_hidden_1 = tf.placeholder("float", [None, 400])
num_hidden_2 = tf.placeholder("float", [None, 100])


# Building the encoder
def encoder(x):

    W1 = tf.get_variable("ew1", dtype=tf.float32, initializer=tf.initializers.truncated_normal(stddev=0.05),shape=[784, 400])
    b1 = tf.get_variable("eb1", dtype=tf.float32, initializer=tf.constant_initializer(0.2), shape=[400])
    layer_1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.get_variable("ew2", dtype=tf.float32, initializer=tf.initializers.truncated_normal(stddev=0.05),shape=[400, 100])
    b2 = tf.get_variable("eb2", dtype=tf.float32, initializer=tf.constant_initializer(0.2), shape=[100])
    layer_2 = tf.nn.relu(tf.matmul(layer_1, W2) + b2)

    return layer_2


# Building the decoder
def decoder(x):
    W1 = tf.get_variable("dw1", dtype=tf.float32, initializer=tf.initializers.truncated_normal(stddev=0.05),shape=[100, 400])
    b1 = tf.get_variable("db1", dtype=tf.float32, initializer=tf.constant_initializer(0.2), shape=[400])
    layer_1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.get_variable("dw2", dtype=tf.float32, initializer=tf.initializers.truncated_normal(stddev=0.05),shape=[400, 784])
    b2 = tf.get_variable("db2", dtype=tf.float32, initializer=tf.constant_initializer(0.2), shape=[784])
    layer_2 = tf.nn.relu(tf.matmul(layer_1, W2) + b2)

    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
batch_size = 256
num_steps = 3000
# Start a new TF session
with tf.Session() as sess:
    # Run the initializer
    sess.run(tf.global_variables_initializer())

    # Training
    for i in range(1, num_steps + 1):
        batch = mnist.train.next_batch(batch_size)

        optimizer.run(feed_dict={X: batch[0], y_true: batch[0]})

        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, 1))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x = mnist[0]
        # Encode and decode the digit image
        g = decoder(encoder(batch_x))

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()