import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import mpld3
import os
import json

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
np.random.seed(0)
tf.set_random_seed(0)


folder = os.path.join(os.path.curdir, 'main')
# os.mkdir(folder)
os.chdir(folder)


class VariationalAutoencoder(object):

    def __init__(self, network_architecture, restore=False, activation_fct=tf.nn.relu,
                 learning_rate=0.001, batch_size=100):
        if restore:
            print("Loading previously trained network")
            self.network_architecture = network_architecture
            self.sess = tf.Session()
            new_saver = tf.train.import_meta_graph(os.path.join(os.path.curdir, 'model', 'model.ckpt.meta'))
            new_saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(os.path.curdir, 'model', './')))
            self.sess.run(tf.global_variables_initializer())
            self._create_loss_optimizer()
            # Launch the session

        else:
            self.network_architecture = network_architecture
            self.activation_fct = activation_fct
            self.learning_rate = learning_rate
            self.batch_size = batch_size

            # tf Graph input
            self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

            # Create autoencoder network
            self._create_network()
            # Define loss function based variational upper-bound and
            # corresponding optimizer
            self._create_loss_optimizer()

            # Initializing the tensor flow variables
            init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.InteractiveSession()
            self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._encoder_network(network_weights["weights_encoder"],
                                      network_weights["biases_encoder"])

        # Draw one sample z from Gaussian distribution
        latent_z = self.network_architecture["latent_z"]
        eps = tf.random_normal((self.batch_size, latent_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._decoder_network(network_weights["weights_decoder"],
                                    network_weights["biases_decoder"])

    def _initialize_weights(self, encoder_1, encoder_2,
                            decoder_1, decoder_2,
                            n_input, latent_z):
        all_weights = dict()
        with tf.variable_scope("weights_encoder"):
            all_weights['weights_encoder'] = {
                'h1': tf.get_variable('h1', shape=[n_input, encoder_1],
                                      initializer=tf.contrib.layers.xavier_initializer()),
                'h2': tf.get_variable('h2', shape=[encoder_1, encoder_2],
                                      initializer=tf.contrib.layers.xavier_initializer()),
                'out_mean': tf.get_variable('out_mean', shape=[encoder_2, latent_z],
                                      initializer=tf.contrib.layers.xavier_initializer()),
                'out_log_sigma': tf.get_variable('out_log_sigma', shape=[encoder_2, latent_z],
                                            initializer=tf.contrib.layers.xavier_initializer())}
        all_weights['biases_encoder'] = {
            'b1': tf.Variable(tf.zeros([encoder_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([encoder_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([latent_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([latent_z], dtype=tf.float32))}
        with tf.variable_scope("weights_decoder"):
            all_weights['weights_decoder'] = {
                'h1': tf.get_variable('h1', shape=[latent_z, decoder_1],
                                            initializer=tf.contrib.layers.xavier_initializer()),
                'h2': tf.get_variable('h2', shape=[decoder_1, decoder_2],
                                            initializer=tf.contrib.layers.xavier_initializer()),
                'out_mean': tf.get_variable('out_mean', shape=[decoder_2, n_input],
                                            initializer=tf.contrib.layers.xavier_initializer()),
                'out_log_sigma': tf.get_variable('out_log_sigma', shape=[decoder_2, n_input],
                                            initializer=tf.contrib.layers.xavier_initializer())}
        all_weights['biases_decoder'] = {
            'b1': tf.Variable(tf.zeros([decoder_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([decoder_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _encoder_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.activation_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.activation_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _decoder_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.activation_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.activation_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Ge_nerate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["latent_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})


def train(model, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=1):
    vae = model
    saver = tf.train.Saver()
    costlist = []
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(avg_cost))
            costlist.append(avg_cost)

        #Visualizations
        path = os.path.join(os.path.curdir, str(epoch))
        # os.mkdir(path)
        #x_sample, _ = mnist.test.next_batch(100)
        #visualize_reconstruction(vae, x_sample, epoch)
        #list_z =[]
        #visualize_latent(vae, x_sample, _, list_z, epoch)
        #visualize_manifold(vae, x_sample, epoch)

    #with open('latent.json', 'a+') as outfile:
     #   json.dump(list_z, outfile)
    #plot cost
    '''thefile = open('final_cost.txt', 'w')
    thefile.write("{}\n{}".format(costlist[0], costlist[-1]))
    plt.plot(costlist)
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.title('cost per epoch')
    plt.savefig('cost_per_epoch')
    plt.tight_layout()
    plt.close()'''

    path = os.path.join(os.path.curdir, 'model')
    #os.mkdir(path)
    #os.chdir(path)
    #saver.save(vae.sess, "model.ckpt")
    return vae


def visualize_latent(model, x_sample, _, list_z, epoch=100, text=True):  # latent must be size 2
    z = model.transform(x_sample)
    '''Data Serialization for D3
    ez = z.tolist()
    c = np.argmax(_, 1)
    c = c.tolist()
    with open('latent-space-data.txt', 'w') as outfile:
        outfile.write("x\t")
        outfile.write("y\t")
        outfile.write("c\n")
        for i in range(0, len(ez)):
            outfile.write("{}\t".format(z[i][0]))
            outfile.write("{}\t".format(z[i][1]))
            outfile.write("{}\n".format(c[i]))
        outfile.close() '''

    '''Bokeh
    list_z.append(z.tolist())
    p = figure(plot_width=400, plot_height=400)
    colors = [palette[x] for x in (np.argmax(_, 1))]
    p.circle(z[:,0], z[:,1], size=20, color=colors, alpha=0.7)
    show(p)'''

    '''MPLD3'''
    f, ax = plt.subplotll_nba_voting/s(1, figsize=(6 * 1.1618, 6))
    im = ax.scatter(z[:,0], z[:,1], c=np.argmax(_, 1), cmap="Vega10",
                    alpha=0.7)
    ax.set_xlabel('First dimension of sampled latent variable $z_1$')
    ax.set_ylabel('Second dimension of sampled latent variable mean $z_2$')
    ax.set_xlim([-4., 4.])
    ax.set_ylim([-4., 4.])
    f.colorbar(im, ax=ax, label='Digit class')
    plt.tight_layout()
    print(mpld3.fig_to_html(f))
    mpld3.show()
    '''END'''
    path = os.path.join(os.path.curdir, str(epoch), 'latent')
    # plt.savefig(path)
    #plt.close()


def visualize_manifold(model, x_sample, epoch=100):
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]] * model.batch_size)
            x_mean = model.generate(z_mu)
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrill_nba_voting/d(x_values, y_values)
    plt.imshow(canvas, origin="upper", cmap="Greys")
    plt.tight_layout()
    #mpld3.show()
    path = os.path.join(os.path.curdir, str(epoch), 'manifold')
    # plt.savefig(path)
    #plt.close()


def visualize_reconstruction(model, x_sample, epoch=100):
    x_reconstruct = model.reconstruct(x_sample)
    plt.figure(figsize=(8, 12))

    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    #mpld3.show()
    path = os.path.join(os.path.curdir, str(epoch), 'reconstruction')
    # plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    network_architecture = \
        dict(encoder_1=500,  # 1st layer encoder neurons
             encoder_2=500,  # 2nd layer encoder neurons
             decoder_1=500,  # 1st layer decoder neurons
             decoder_2=500,  # 2nd layer decoder neurons
             n_input=784,  # MNIST data input (img shape: 28*28)
             latent_z=2)  # dimensionality of latent space

    vae = VariationalAutoencoder(network_architecture)
    train(vae, training_epochs=100)
    x_sample, _ = mnist.test.next_batch(100)
    list_z =[]
    visualize_latent(vae, x_sample, _, list_z)
