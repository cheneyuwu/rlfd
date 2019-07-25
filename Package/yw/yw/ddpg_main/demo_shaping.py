import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from yw.util.tf_util import NormalizingFlow, MAF


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

    def reward(self, o, g, u, o_2, g_2, u_2):
        potential = self.potential(o, g, u)
        next_potential = self.potential(o_2, g_2, u_2)
        assert potential.shape[1] == next_potential.shape[1] == 1
        return self.gamma * next_potential - potential

    def potential(self, o, g, u):
        raise NotImplementedError

    def save_weights(self, sess, path):
        if self.saver:
            self.saver.save(sess, path)

    def load_weights(self, sess, path):
        if self.saver:
            self.saver.restore(sess, path)

    def _concat_inputs(self, o, g, u):
        # Concat demonstration inputs
        state_tf = o
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, g])
        state_tf = tf.concat(axis=1, values=[state_tf, u])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf

    def _cast_concat_inputs(self, o, g, u):
        # Concat demonstration inputs
        state_tf = tf.cast(o, tf.float64)
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(g, tf.float64)])
        state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(u, tf.float64)])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf


class ManualDemoShaping(DemoShaping):
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
        super().__init__(gamma)

    def potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        # state based, for reacher 1d, and 2d
        ret = -tf.norm(tf.abs(o - g), ord=2, axis=1, keepdims=True) * 10
        # state action based for reacher 1d
        # ret = -(tf.clip_by_value((g - o) * 10, -2, 2) - u) ** 2
        return ret


class GaussianDemoShaping(DemoShaping):
    def __init__(self, gamma, demo_inputs_tf):
        """
        Implement the state, action based potential function and corresponding actions
        Args:
            o - observation
            g - goal
            u - action
            o_2 - observation that goes to
            g_2 - same as g, idk why I must make this specific
            u_2 - output from the actor of the main network
            demo_inputs_tf - demo_inputs that contains all the transitons from demonstration
        """
        self.demo_inputs_tf = demo_inputs_tf
        self.demo_state_tf = self._concat_inputs(
            self.demo_inputs_tf["o"],
            self.demo_inputs_tf["g"] if "g" in self.demo_inputs_tf.keys() else None,
            self.demo_inputs_tf["u"],
        )
        self.sigma = 1.0  # a hyperparam to be tuned
        self.scale = 1.0  # another hyperparam to be tuned
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


class NormalizingFlowDemoShaping(DemoShaping):
    def __init__(self, gamma, demo_inputs_tf):
        """
        Implement the state, action based potential function and corresponding actions
        Args:
            o - observation
            g - goal
            u - action
            o_2 - observation that goes to
            g_2 - same as g, idk why I must make this specific
            u_2 - output from the actor of the main network
            demo_inputs_tf - demo_inputs that contains all the transitons from demonstration
        """
        self.demo_inputs_tf = demo_inputs_tf
        demo_state_tf = self._concat_inputs(
            self.demo_inputs_tf["o"],
            self.demo_inputs_tf["g"] if "g" in self.demo_inputs_tf.keys() else None,
            self.demo_inputs_tf["u"],
        )

        demo_state_dim = int(demo_state_tf.shape[1])
        self.base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([demo_state_dim], tf.float32))
        # normalizing flow nn
        self.nn = NormalizingFlow(base_dist=self.base_dist, d=demo_state_dim, r=demo_state_dim, num_layers=10)
        # loss function that tries to maximize log prob
        self.loss = -tf.reduce_mean(self.nn(demo_state_tf))
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

        super().__init__(gamma)

    def potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        state_tf = self._concat_inputs(o, g, u)

        return self.nn(state_tf)


class MAFDemoShaping(DemoShaping):
    def __init__(self, gamma, demo_inputs_tf, prm_loss_weight, reg_loss_weight, potential_weight):
        """
        Args:
            o - observation
            g - goal
            u - action
            o_2 - observation that goes to
            g_2 - same as g, idk why I must make this specific
            u_2 - output from the actor of the main network
            demo_inputs_tf - demo_inputs that contains all the transitons from demonstration
        """

        self.demo_inputs_tf = demo_inputs_tf
        demo_state_tf = self._cast_concat_inputs(
            self.demo_inputs_tf["o"],
            self.demo_inputs_tf["g"] if "g" in self.demo_inputs_tf.keys() else None,
            self.demo_inputs_tf["u"],
        )

        # normalizing flow nn
        demo_state_dim = int(demo_state_tf.shape[1])
        self.base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([demo_state_dim], tf.float64))
        self.nn = MAF(base_dist=self.base_dist, dim=demo_state_dim, num_layers=6)

        # loss function that tries to maximize log prob
        # log probability
        neg_log_prob = -tf.reduce_mean(tf.reshape(self.nn.log_prob(demo_state_tf), (-1, 1)))
        # regularizer
        jacobian = tf.gradients(neg_log_prob, demo_state_tf)
        regularizer = tf.norm(jacobian[0], ord=2)
        self.loss = prm_loss_weight * neg_log_prob + reg_loss_weight * regularizer
        # optimizers
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # params for potentials
        self.scale = tf.constant(5, dtype=tf.float64)
        self.weight = tf.constant(potential_weight, dtype=tf.float64)

        super().__init__(gamma)

    def potential(self, o, g, u):
        """
        Just return negative value of distance between current state and goal state
        """
        state_tf = self._cast_concat_inputs(o, g, u)

        # method 1 with shift and annealing
        potential = tf.reshape(self.nn.prob(state_tf), (-1, 1))
        potential = tf.log(potential + tf.exp(-self.scale))
        potential = potential + self.scale  # add shift
        potential = self.weight * potential / self.scale  # add scaling

        # method 2 add noise to the state and take avg
        # scale = 1000.0
        # num_sample = 8
        # expanded_state_tf = tf.tile(tf.expand_dims(state_tf, 1), [1, num_sample, 1])
        # # make noise
        # noise_dist = tfd.Normal(loc=[0.0] * expanded_state_tf.shape[-1], scale=0.04)
        # noise_tf = tf.cast(noise_dist.sample(tf.shape(expanded_state_tf)[:-1]), tf.float64)
        # noise_tf = tf.clip_by_value(noise_tf, -0.04, 0.04)
        # expanded_state_with_noise_tf = expanded_state_tf + noise_tf
        # potential = tf.reshape(self.nn.log_prob(expanded_state_with_noise_tf), (-1, num_sample, 1))
        # potential = tf.clip_by_value(potential, -scale, 1000000.0)
        # potential = potential + scale  # add shift
        # potential = potential / scale  # add scaling
        # potential = tf.reduce_mean(potential, axis=1)

        return tf.cast(potential, tf.float32)


# Testing
# =====================================

import numpy as np
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


def test_nf(demo_file, **kwargs):

    batch_size = 512

    def sample_target_halfmoon():
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

    def visualize_flow(gs, row, samples, titles):
        X0 = samples[0]

        for i, j in zip([0, len(samples) - 1], [0, 1]):  # range(len(samples)):
            X1 = samples[i]

            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
            ax = pl.subplot(gs[row, j])
            ax.clear()
            pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color="red")

            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color="green")

            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color="blue")

            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color="black")
            pl.xlim([-5, 30])
            pl.ylim([-10, 10])
            # pl.xlim([-1, 1])
            # pl.ylim([-1, 1])
            pl.xlabel("state")
            pl.ylabel("action")
            pl.title(titles[j])

    tf.set_random_seed(0)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    if demo_file == None:
        x_samples = sample_target_halfmoon()
        np_samples = sess.run(x_samples)
    else:
        demo_data = np.load(demo_file)  # load the demonstration data from data file
        # reach1d first order
        # o = demo_data["o"][:, :-1, :].reshape((-1, 1))
        # u = demo_data["u"].reshape((-1, 1))
        # reach1d second order
        o = demo_data["o"][:, :-1, :].reshape((-1, 2))[:, 0:1]
        u = demo_data["o"][:, :-1, :].reshape((-1, 2))[:, 1:2]
        np_samples = np.concatenate((o, u), axis=1)

    # Turn on interactive mode to see immediate change
    pl.ion()

    pl.figure()
    gs = gridspec.GridSpec(3, 2)

    # Training dataset
    ax = pl.subplot(gs[0, 0])
    ax.clear()
    ax.scatter(np_samples[:, 0], np_samples[:, 1], s=10, color="red")
    pl.xlim([-5, 30])
    pl.ylim([-10, 10])
    # pl.xlim([-1, 1])
    # pl.ylim([-1, 1])
    pl.xlabel("state")
    pl.ylabel("action")
    pl.title("Training samples")

    inputs_tf = {}
    inputs_tf["o"] = tf.placeholder(shape=(None, 1), dtype=tf.float64)
    inputs_tf["u"] = tf.placeholder(shape=(None, 1), dtype=tf.float64)
    # demo_shaping = NormalizingFlowDemoShaping(gamma=1.0, demo_inputs_tf=inputs_tf)
    demo_shaping = MAFDemoShaping(gamma=1.0, demo_inputs_tf=inputs_tf)

    sess.run(tf.global_variables_initializer())

    # Flow before training
    _, samples_no_training, _ = sample_flow(demo_shaping.base_dist, demo_shaping.nn.dist)
    samples_no_training = sess.run(samples_no_training)
    visualize_flow(gs, 1, samples_no_training, ["Base dist", "Samples w/o training"])

    for i in range(40000):
        if demo_file == None:
            np_samples = sess.run(x_samples)
            feed = {inputs_tf["o"]: np_samples[:, 0:1], inputs_tf["u"]: np_samples[:, 1:]}
        else:
            feed = {inputs_tf["o"]: o, inputs_tf["u"]: u}
        _, np_loss = sess.run([demo_shaping.train_op, demo_shaping.loss], feed_dict=feed)
        if i % int(100) == 0:
            print(i, np_loss)

        if i % int(100) == 0:
            # Flow after training
            _, samples_with_training, _ = sample_flow(demo_shaping.base_dist, demo_shaping.nn.dist)
            samples_with_training = sess.run(samples_with_training)
            visualize_flow(gs, 2, samples_with_training, ["Base dist", "Samples w/ training"])

            # Check potential function
            potential = demo_shaping.potential(inputs_tf["o"], None, inputs_tf["u"])

            num_point = 24
            ls = np.linspace(-1.0, 1.0, num_point)
            ls2 = ls  # * 2
            o_test, u_test = np.meshgrid(ls, ls2)
            o_r = o_test.reshape((-1, 1))
            u_r = u_test.reshape((-1, 1))

            feed = {inputs_tf["o"]: o_r, inputs_tf["u"]: u_r}
            ret = sess.run([potential], feed_dict=feed)

            o_r = o_test.reshape((num_point, num_point))
            u_r = u_test.reshape((num_point, num_point))
            for i in range(len(ret)):
                ret[i] = ret[i].reshape((num_point, num_point))

            ax = pl.subplot(gs[0, 1], projection="3d")
            ax.clear()
            ax.plot_surface(o_r, u_r, ret[0])
            # ax.set_zlim(0, 3)
            ax.set_xlabel("observation")
            ax.set_ylabel("action")
            ax.set_zlabel("potential")
            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
                + ax.get_zticklabels()
            ):
                item.set_fontsize(6)

            pl.show()
            pl.pause(0.001)


if __name__ == "__main__":

    import sys
    from yw.util.cmd_util import ArgParser

    ap = ArgParser()
    ap.parser.add_argument("--demo_file", help="demonstration file", type=str, default=None)
    ap.parse(sys.argv)

    test_nf(**ap.get_dict())
