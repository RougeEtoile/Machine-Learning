""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10
beta = 0.01

# Network Parameters
latent_z = 10  # compressed reprensentation
n_hidden_1 = 500  # 1st layer num features
n_hidden_2 = 500  # 2nd layer num features
n_input = 784  # MNvim.wikia.com/wiki/Search_and_replaceIST data input (img shape: 28*28)

# tf Graph input (only pictures)
x = tf.placeholder("float", [None, n_input])
y_ = tf.placeholder("float", [None, 10])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_z': tf.Variable(tf.random_normal([n_hidden_2, latent_z])),
    'decoder_z': tf.Variable(tf.random_normal([latent_z, n_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
    'encoder_zb': tf.Variable(tf.random_normal([latent_z])),
    'decoder_zb': tf.Variable(tf.random_normal([n_hidden_1])),
}

# Building the encoder


def encoder(x):
    # Encoder Hidden layer with relu activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with relu activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    latent_layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_z']),
                                   biases['encoder_zb']))
    return latent_layer
# Building the decoder


def decoder(x):
    # Decoder Hidden layer with relu activation #1
    print("decode")
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_z']), biases['decoder_zb']))
    # Decoder Hidden layer with relu activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h1']),
                                   biases['decoder_b1']))
    latent_layer = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_h2']),
                                        biases['decoder_b2']))
    return latent_layer

# Construct model
encoder_op = encoder(x)
# decoder_op = decoder(encoder_op)

# Prediction
y = encoder_op
# Tar

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
# plot train error
trainError = []
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(batch_size):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            c = sess.run([cost, optimizer], feed_dict={x: batch_xs, y_: batch_ys})
        if epoch % display_step == 0:
            trainError.append(c)    #
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", c[0])
        #trainError.append(c)
    print("Optimization Finished!")
    #print(trainError)
    # plt.plot(range(training_epochs), trainError, "ro")
    # plt.show()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

    # Applying encode and decode over test set

    """encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    print(len(trainError))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
"""