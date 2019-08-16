import itertools

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from yw.util.tf_util import MAF, RealNVP, MLP


class DemoShaping:
    def __init__(self, gamma):
        """
        Implement the state, action based potential function and corresponding actions
        Args:
            o - observation
            g - goal
            u - action
            o_2 - observation that goes to
            g_2 - same as g, idk why I must make this specific
            u_2 - output from the actor of the main network
        """
        self.gamma = gamma
        self.scope = tf.get_variable_scope()
        if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name):
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name))
        else:
            self.saver = None

    def potential(self, o, g, u):
        raise NotImplementedError

    def reward(self, o, g, u, o_2, g_2, u_2):
        potential = self.potential(o, g, u)
        next_potential = self.potential(o_2, g_2, u_2)
        assert potential.shape[1] == next_potential.shape[1] == 1
        return self.gamma * next_potential - potential

    def save_weights(self, sess, path):
        if self.saver:
            self.saver.save(sess, path)

    def load_weights(self, sess, path):
        if self.saver:
            self.saver.restore(sess, path)

    def _concat_inputs(self, o, g, u):
        # concat demonstration inputs
        state_tf = o
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, g])
        state_tf = tf.concat(axis=1, values=[state_tf, u])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf

    def _cast_concat_inputs(self, o, g, u):
        # concat demonstration inputs
        state_tf = tf.cast(o, tf.float64)
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(g, tf.float64)])
        state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(u, tf.float64)])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf


class GaussianDemoShaping(DemoShaping):
    def __init__(self, gamma, demo_inputs_tf):
        """
        Implement the state, action based potential function and corresponding actions
        """
        self.demo_inputs_tf = demo_inputs_tf
        self.demo_state_tf = self._concat_inputs(
            self.demo_inputs_tf["o"],
            self.demo_inputs_tf["g"] if "g" in self.demo_inputs_tf.keys() else None,
            self.demo_inputs_tf["u"],
        )
        self.sigma = 1.0  # hyperparam to be tuned
        self.scale = 1.0  # hyperparam to be tuned
        super().__init__(gamma)

    def potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        state_tf = self._concat_inputs(o, g, u)

        # Calculate the potential
        # expand dimension of demo_state and state so that they have the same shape: (batch_size, num_demo, k)
        expanded_demo_state_tf = tf.tile(tf.expand_dims(self.demo_state_tf, 0), [tf.shape(state_tf)[0], 1, 1])
        expanded_state_tf = tf.tile(tf.expand_dims(state_tf, 1), [1, tf.shape(self.demo_state_tf)[0], 1])
        # calculate distance, result shape is (batch_size, num_demo, k)
        distance_tf = expanded_state_tf - expanded_demo_state_tf
        # calculate L2 Norm square, result shape is (batch_size, num_demo)
        norm_tf = tf.norm(distance_tf, ord=2, axis=-1)
        # cauculate multi var gaussian, result shape is (batch_size, num_demo)
        gaussian_tf = tf.exp(-0.5 * self.sigma * norm_tf * norm_tf)  # let sigma be 5
        # sum the result from all demo transitions and get the final result, shape is (batch_size, 1)
        potential = self.scale * tf.reduce_max(gaussian_tf, axis=-1, keepdims=True)

        return potential


class NFDemoShaping(DemoShaping):
    def __init__(
        self,
        gamma,
        demo_inputs_tf,
        nf_type,
        lr,
        num_masked,
        num_bijectors,
        layer_sizes,
        initializer_type,
        prm_loss_weight,
        reg_loss_weight,
        potential_weight,
    ):
        """
        Args:
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            prm_loss_weight  (float)
            reg_loss_weight  (float)
            potential_weight (float)
        """
        self.demo_inputs_tf = demo_inputs_tf
        demo_state_tf = self._cast_concat_inputs(
            self.demo_inputs_tf["o"],
            self.demo_inputs_tf["g"] if "g" in self.demo_inputs_tf.keys() else None,
            self.demo_inputs_tf["u"],
        )
        # params for potentials
        self.scale = tf.constant(5, dtype=tf.float64)
        self.potential_weight = tf.constant(potential_weight, dtype=tf.float64)
        # normalizing flow nn
        demo_state_dim = int(demo_state_tf.shape[1])
        self.base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([demo_state_dim], tf.float64))
        if nf_type == "maf":
            self.nf = MAF(
                base_dist=self.base_dist,
                dim=demo_state_dim,
                num_bijectors=num_bijectors,
                layer_sizes=layer_sizes,
                initializer_type=initializer_type,
            )
        elif nf_type == "realnvp":
            self.nf = RealNVP(
                base_dist=self.base_dist,
                dim=demo_state_dim,
                num_masked=num_masked,
                num_bijectors=num_bijectors,
                layer_sizes=layer_sizes,
                initializer_type=initializer_type,
            )
        else:
            assert False
        # loss function that tries to maximize log prob
        # log probability
        neg_log_prob = tf.clip_by_value(-self.nf.log_prob(demo_state_tf), -1e5, 1e5)
        neg_log_prob = tf.reduce_mean(tf.reshape(neg_log_prob, (-1, 1)))
        # regularizer
        jacobian = tf.gradients(neg_log_prob, demo_state_tf)
        regularizer = tf.norm(jacobian[0], ord=2)
        self.loss = prm_loss_weight * neg_log_prob + reg_loss_weight * regularizer
        # optimizers
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        super().__init__(gamma)

    def potential(self, o, g, u):
        state_tf = self._cast_concat_inputs(o, g, u)

        potential = tf.reshape(self.nf.prob(state_tf), (-1, 1))
        potential = tf.log(potential + tf.exp(-self.scale))
        potential = potential + self.scale  # add shift
        potential = self.potential_weight * potential / self.scale  # add scaling
        return tf.cast(potential, tf.float32)

    def train(self, sess, feed_dict={}):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss


class EnsNFDemoShaping(DemoShaping):
    def __init__(
        self,
        num_ens,
        gamma,
        demo_inputs_tf,
        nf_type,
        lr,
        num_masked,
        num_bijectors,
        layer_sizes,
        initializer_type,
        prm_loss_weight,
        reg_loss_weight,
        potential_weight,
    ):
        """
        Args:
            num_ens          (int)   - number of nf ensembles
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            prm_loss_weight  (float)
            reg_loss_weight  (float)
            potential_weight (float)
        """
        # setup ensemble
        self.nfs = []
        for i in range(num_ens):
            with tf.variable_scope("ens_" + str(i)):
                self.nfs.append(
                    NFDemoShaping(
                        nf_type=nf_type,
                        gamma=gamma,
                        lr=lr,
                        num_masked=num_masked,
                        num_bijectors=num_bijectors,
                        layer_sizes=layer_sizes,
                        initializer_type=initializer_type,
                        demo_inputs_tf=demo_inputs_tf,
                        prm_loss_weight=prm_loss_weight,
                        reg_loss_weight=reg_loss_weight,
                        potential_weight=potential_weight,
                    )
                )
        # loss
        self.loss = tf.reduce_sum([ens.loss for ens in self.nfs], axis=0)
        # optimizers
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        super().__init__(gamma)

    def potential(self, o, g, u):
        # return the mean potential of all ens
        potential = tf.reduce_mean([ens.potential(o=o, g=g, u=u) for ens in self.nfs], axis=0)
        assert potential.shape[1] == 1
        return potential

    def train(self, sess, feed_dict={}):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss


class GANDemoShaping(DemoShaping):
    def __init__(
        self,
        gamma,
        demo_inputs_tf,
        potential_weight,
        layer_sizes,
        initializer_type,
        latent_dim,
        gp_lambda,
        critic_iter,
    ):
        """
        GAN with Wasserstein distance plus gradient penalty.
        Args:
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            potential_weight (float)
        """
        # Parameters
        self.demo_inputs_tf = demo_inputs_tf
        demo_state_tf = self._concat_inputs(
            self.demo_inputs_tf["o"],
            self.demo_inputs_tf["g"] if "g" in self.demo_inputs_tf.keys() else None,
            self.demo_inputs_tf["u"],
        )
        self.potential_weight = potential_weight
        self.critic_iter = critic_iter
        self.train_gen = 0  # counter

        # Generator & Discriminator
        with tf.variable_scope("generator"):
            input_shape = (None, latent_dim)  # latent space dimensions
            gen_output_dim = demo_state_tf.shape[-1]
            self.generator = MLP(
                input_shape=input_shape,
                layers_sizes=layer_sizes + [demo_state_tf.shape[-1]],
                initializer_type=initializer_type,
            )
        with tf.variable_scope("discriminator"):
            input_shape = (None, demo_state_tf.shape[-1])
            self.discriminator = MLP(
                input_shape=input_shape, layers_sizes=layer_sizes + [1], initializer_type=initializer_type
            )

        # Loss functions
        assert len(demo_state_tf.shape) == 2
        fake_data = self.generator(tf.random.normal([tf.shape(demo_state_tf)[0], latent_dim]))
        self.fake_data = fake_data  # exposed for internal testing
        disc_fake = self.discriminator(fake_data)
        disc_real = self.discriminator(demo_state_tf)
        # discriminator loss (including gp loss)
        alpha = tf.random_uniform(shape=[tf.shape(demo_state_tf)[0], 1], minval=0.0, maxval=1.0)
        interpolates = alpha * demo_state_tf + (1.0 - alpha) * fake_data
        disc_interpolates = self.discriminator(interpolates)
        gradients = tf.gradients(disc_interpolates, [interpolates][0])
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
        self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) + gp_lambda * gradient_penalty
        # generator loss
        self.gen_cost = -tf.reduce_mean(disc_fake)

        # Training
        self.disc_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + "/" + "discriminator"
        )
        self.gen_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + "/" + "generator"
        )
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.disc_cost, var_list=self.disc_vars
        )
        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.gen_cost, var_list=self.gen_vars
        )

        super().__init__(gamma)

    def potential(self, o, g, u):
        """
        Use the output of the GAN's discriminator as potential.
        """
        state_tf = self._concat_inputs(o, g, u)
        potential = self.discriminator(state_tf)
        potential = potential * self.potential_weight
        return potential

    def train(self, sess, feed_dict={}):
        # train critic
        disc_cost, _ = sess.run([self.disc_cost, self.disc_train_op], feed_dict=feed_dict)
        # train generator
        if self.train_gen == 0:
            sess.run(self.gen_train_op)
        self.train_gen = np.mod(self.train_gen + 1, self.critic_iter)
        return disc_cost


class EnsGANDemoShaping(DemoShaping):
    def __init__(
        self,
        num_ens,
        gamma,
        demo_inputs_tf,
        layer_sizes,
        initializer_type,
        latent_dim,
        gp_lambda,
        critic_iter,
        potential_weight,
    ):
        """
        Args:
            num_ens          (int)   - number of ensembles
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            potential_weight (float)
        """
        # Parameters for training
        self.critic_iter = critic_iter
        self.train_gen = 0  # counter

        # Setup ensemble
        self.gans = []
        for i in range(num_ens):
            with tf.variable_scope("ens_" + str(i)):
                self.gans.append(
                    GANDemoShaping(
                        gamma=gamma,
                        demo_inputs_tf=demo_inputs_tf,
                        layer_sizes=layer_sizes,
                        initializer_type=initializer_type,
                        latent_dim=latent_dim,
                        gp_lambda=gp_lambda,
                        critic_iter=critic_iter,
                        potential_weight=potential_weight,
                    )
                )

        # Loss functions
        self.disc_cost = tf.reduce_sum([ens.disc_cost for ens in self.gans], axis=0)
        self.gen_cost = tf.reduce_sum([ens.gen_cost for ens in self.gans], axis=0)

        # Training
        self.disc_vars = list(itertools.chain(*[ens.disc_vars for ens in self.gans]))
        self.gen_vars = list(itertools.chain(*[ens.gen_vars for ens in self.gans]))
        self.disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.disc_cost, var_list=self.disc_vars
        )
        self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.gen_cost, var_list=self.gen_vars
        )

        super().__init__(gamma)

    def potential(self, o, g, u):
        # return the mean potential of all ens
        potential = tf.reduce_mean([ens.potential(o=o, g=g, u=u) for ens in self.gans], axis=0)
        assert potential.shape[1] == 1
        return potential

    def train(self, sess, feed_dict={}):
        # train critic
        disc_cost, _ = sess.run([self.disc_cost, self.disc_train_op], feed_dict=feed_dict)
        # train generator
        if self.train_gen == 0:
            sess.run(self.gen_train_op)
        self.train_gen = np.mod(self.train_gen + 1, self.critic_iter)
        return disc_cost


# Testing
# =====================================

import random
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def test_nf():
    def sample_target_halfmoon(batch_size=512):
        x2_dist = tfd.Normal(loc=tf.cast(0.0, tf.float32), scale=tf.cast(4.0, tf.float32))
        x2_samples = x2_dist.sample(batch_size)
        x1 = tfd.Normal(loc=0.25 * tf.square(x2_samples), scale=tf.ones(batch_size, dtype=tf.float32))
        x1_samples = x1.sample()
        x_samples = tf.stack([x1_samples, x2_samples], axis=1)
        return x_samples  # shape (batch_size, 2)

    def sample_flow(base_dist, transformed_dist):
        x = base_dist.sample(512)
        samples = [x]
        names = [base_dist.name]
        for bijector in reversed(transformed_dist.bijector.bijectors):
            x = bijector.forward(x)
            samples.append(x)
            names.append(bijector.name)

        return x, samples, names

    def visualize_flow(axs, samples, titles):
        X0 = samples[0]

        for i, j in zip([0, len(samples) - 1], [0, 1]):  # range(len(samples)):
            X1 = samples[i]

            ax = axs[j]
            ax.clear()

            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
            ax.scatter(X1[idx, 0], X1[idx, 1], s=10, color="red")

            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            ax.scatter(X1[idx, 0], X1[idx, 1], s=10, color="green")

            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            ax.scatter(X1[idx, 0], X1[idx, 1], s=10, color="blue")

            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            ax.scatter(X1[idx, 0], X1[idx, 1], s=10, color="black")

            ax.set_xlabel("s1")
            ax.set_ylabel("s2")
            ax.set_xlim([-5, 30])
            ax.set_ylim([-10, 10])
            ax.set_title(titles[j])

    tf.set_random_seed(0)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    x_samples = sample_target_halfmoon()
    np_samples = sess.run(x_samples)

    pl.ion()  # turn on interactive mode to see immediate change
    pl.figure()
    gs = gridspec.GridSpec(3, 2)
    # training dataset plot
    ax = pl.subplot(gs[0, 0])
    ax.clear()
    ax.scatter(np_samples[:, 0], np_samples[:, 1], s=10, color="red")
    ax.set_xlim([-5, 30])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    ax.set_title("Training samples")
    # ax for plotting the contour map
    ct_ax = pl.subplot(gs[0, 1])
    ct_cax = make_axes_locatable(ct_ax).append_axes("right", size="5%", pad="2%")
    # ax for visualizing flow wo training
    vf_axs_wo_training = [pl.subplot(gs[1, j]) for j in range(2)]
    # ax for visualizing flow with training
    vf_axs_w_training = [pl.subplot(gs[2, j]) for j in range(2)]

    inputs_tf = {}
    inputs_tf["o"] = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    inputs_tf["u"] = tf.placeholder(shape=(None, 1), dtype=tf.float32)

    demo_shaping = NFDemoShaping(
        gamma=1.0,
        demo_inputs_tf=inputs_tf,
        nf_type="maf",  # choose between maf and realnvp
        lr=5e-4,
        num_masked=1,
        num_bijectors=6,
        layer_sizes=[512, 512],
        prm_loss_weight=1.0,
        reg_loss_weight=5e3,
        potential_weight=1.0,
    )

    sess.run(tf.global_variables_initializer())

    # Flow before training
    _, samples_no_training, _ = sample_flow(demo_shaping.base_dist, demo_shaping.nf.dist)
    samples_no_training = sess.run(samples_no_training)
    visualize_flow(vf_axs_wo_training, samples_no_training, ["Base dist", "Samples w/o training"])

    for i in range(5000):
        np_samples = sess.run(x_samples)
        feed = {inputs_tf["o"]: np_samples[:, 0:1], inputs_tf["u"]: np_samples[:, 1:]}
        np_loss = demo_shaping.train(sess=sess, feed_dict=feed)
        if i % int(100) == 0:
            print(i, np_loss)

        if i % int(100) == 0:
            # Flow after training
            _, samples_with_training, _ = sample_flow(demo_shaping.base_dist, demo_shaping.nf.dist)
            samples_with_training = sess.run(samples_with_training)
            visualize_flow(vf_axs_w_training, samples_with_training, ["Base dist", "Samples w/ training"])

            # Check potential function
            potential = demo_shaping.potential(inputs_tf["o"], None, inputs_tf["u"])

            num_point = 100
            ls = np.linspace(-100.0, 100.0, num_point)
            o, u = np.meshgrid(ls, ls)
            o_r = o.reshape((-1, 1))
            u_r = u.reshape((-1, 1))

            feed = {inputs_tf["o"]: o_r, inputs_tf["u"]: u_r}
            ret = sess.run(potential, feed_dict=feed)
            ret = ret.reshape((num_point, num_point))

            ct_ax.clear()
            ct = ct_ax.contour(o, u, ret)
            pl.colorbar(ct, cax=ct_cax)
            ct_ax.set_xlim([-5, 30])
            ct_ax.set_ylim([-10, 10])
            ct_ax.set_xlabel("o_1")
            ct_ax.set_ylabel("o_2")

            pl.show()
            pl.pause(1.0)


def test_gan():

    # Dataset iterator
    def inf_train_gen(batch_size):
        # generate mix of gaussian with 8 different mean, same variance
        scale = 2.0
        centers = [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]

        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            batch_data = []
            for _ in range(batch_size):
                point = np.random.randn(2) * 0.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                batch_data.append(point)

            batch_data = np.array(batch_data, dtype=np.float32)
            batch_data /= 1.414  # std
            yield batch_data

    batch_size = 128
    inputs_tf = {}
    inputs_tf["o"] = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    inputs_tf["u"] = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    with tf.variable_scope("shaping"):
        gan_demo_shaping = GANDemoShaping(
            gamma=1.0,
            demo_inputs_tf=inputs_tf,
            batch_size=batch_size,
            layer_sizes=[256, 256],
            latent_dim=2,
            gp_lambda=0.1,
            critic_iter=5,
        )
        potential = gan_demo_shaping.potential(o=inputs_tf["o"], g=None, u=inputs_tf["u"])

    plt.ion()

    # Train loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data_gen = inf_train_gen(batch_size)

        for idx in range(1000000):

            batch_data = data_gen.__next__()

            gan_demo_shaping.train(
                sess=sess,
                batch_data=batch_data,
                feed_dict={inputs_tf["o"]: batch_data[..., 0:1], inputs_tf["u"]: batch_data[..., 1:2]},
            )

            if (np.mod(idx, 1000) == 0) or (idx + 1 == 1000000):

                N_POINTS = 128
                RANGE = 2

                points = np.zeros((N_POINTS, N_POINTS, 2), dtype="float32")
                points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
                points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
                points = points.reshape((-1, 2))

                samples, disc_map = sess.run(
                    [gan_demo_shaping.fake_data, potential],
                    feed_dict={inputs_tf["o"]: points[..., 0:1], inputs_tf["u"]: points[..., 1:2]},
                )

                plt.clf()
                x = y = np.linspace(-RANGE, RANGE, N_POINTS)
                plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
                plt.colorbar()  # add color bar

                plt.scatter(batch_data[:, 0], batch_data[:, 1], c="orange", marker="+")
                plt.scatter(samples[:, 0], samples[:, 1], c="green", marker="*")

                plt.show()
                plt.pause(1)


if __name__ == "__main__":
    test_nf()
    test_gan()
