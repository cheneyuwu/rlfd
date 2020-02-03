import itertools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from td3fd.ddpg.normalizer import Normalizer
from td3fd import logger
from td3fd.memory import iterbatches
from td3fd.ddpg.normalizing_flows import MAF, RealNVP
from td3fd.ddpg.gan_network import Generator, Discriminator

tfd = tfp.distributions


class Shaping:
    def __init__(self, gamma):
        """
        Implement the state, action based potential function and corresponding actions
        Args:
            o - observation
            g - goal
            u - action
            o_2 - observation that goes to
            g_2 - same as g
            u_2 - output from the actor of the main network
        """
        self.gamma = gamma

    def potential(self, o, g, u):
        raise NotImplementedError

    def reward(self, o, g, u, o_2, g_2, u_2):
        potential = self.potential(o, g, u)
        next_potential = self.potential(o_2, g_2, u_2)
        assert potential.shape[1] == next_potential.shape[1] == 1
        return self.gamma * next_potential - potential

    def train(self, batch):
        pass

    def evaluate(self, batch):
        pass

    def post_training_update(self, batch):
        """
        """
        pass

    def _concat_normalize_inputs(self, o, g, u):
        # concat demonstration inputs
        state_tf = self.o_stats.normalize(o)
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, self.g_stats.normalize(g)])
        state_tf = tf.concat(axis=1, values=[state_tf, u / self.max_u])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf

    def _concat_inputs(self, o, g, u):
        # concat demonstration inputs
        state_tf = o
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, g])
        state_tf = tf.concat(axis=1, values=[state_tf, u])
        # note: shape of state_tf is (num_demo, k), where k is sum of dim o g u
        return state_tf

    def _cast_concat_normalize_inputs(self, o, g, u):
        # concat demonstration inputs
        state_tf = tf.cast(self.o_stats.normalize(o), tf.float64)
        if g != None:
            # for multigoal environments, we have goal as another states
            state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(self.g_stats.normalize(g), tf.float64)])
        state_tf = tf.concat(axis=1, values=[state_tf, tf.cast(u / self.max_u, tf.float64)])
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


class NFShaping(Shaping):
    def __init__(
        self,
        sess,
        dims,
        gamma,
        max_u,
        num_bijectors,
        layer_sizes,
        num_masked,
        potential_weight,
        norm_obs,
        norm_eps,
        norm_clip,
        prm_loss_weight,
        reg_loss_weight,
    ):
        """
        Args:
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            prm_loss_weight  (float)
            reg_loss_weight  (float)
            potential_weight (float)
        """
        super().__init__(gamma)

        # Prepare parameters
        self.sess = sess
        self.dims = dims
        self.dimo = self.dims["o"]
        self.dimg = self.dims["g"]
        self.dimu = self.dims["u"]
        self.max_u = max_u
        self.num_bijectors = num_bijectors
        self.layer_sizes = layer_sizes
        self.norm_obs = norm_obs
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.prm_loss_weight = prm_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.potential_weight = tf.constant(potential_weight, dtype=tf.float64)

        #
        self.learning_rate = 2e-4 # TODO this is different from the td3 implementation, check this!
        self.scale = tf.constant(5, dtype=tf.float64)


        self.demo_inputs_tf = {}
        self.demo_inputs_tf["o"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimo))
        self.demo_inputs_tf["u"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimu))
        if self.dimg != (0,):
            self.demo_inputs_tf["g"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimg))

        # normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # training
        demo_state_tf = self._cast_concat_normalize_inputs(
            self.demo_inputs_tf["o"],
            self.demo_inputs_tf["g"] if "g" in self.demo_inputs_tf.keys() else None,
            self.demo_inputs_tf["u"],
        )

        # normalizing flow
        demo_state_dim = int(demo_state_tf.shape[1])
        self.base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([demo_state_dim], tf.float64))
        self.nf = MAF(
            base_dist=self.base_dist, dim=demo_state_dim, num_bijectors=num_bijectors, layer_sizes=layer_sizes
        )
        # loss function that tries to maximize log prob
        # log probability
        neg_log_prob = tf.clip_by_value(-self.nf.log_prob(demo_state_tf), -1e5, 1e5)
        neg_log_prob = tf.reduce_mean(tf.reshape(neg_log_prob, (-1, 1)))
        # regularizer
        jacobian = tf.gradients(neg_log_prob, demo_state_tf)
        regularizer = tf.norm(jacobian[0], ord=2)
        self.loss = prm_loss_weight * neg_log_prob + reg_loss_weight * regularizer
        # optimizers
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def potential(self, o, g, u):
        state_tf = self._cast_concat_normalize_inputs(o, g, u)

        potential = tf.reshape(self.nf.prob(state_tf), (-1, 1))
        potential = tf.math.log(potential + tf.exp(-self.scale))
        potential = potential + self.scale  # shift
        potential = self.potential_weight * potential / self.scale  # scale
        return tf.cast(potential, tf.float32)

    def update_stats(self, batch):
        # add transitions to normalizer
        if not self.norm_obs:
            return
        self.o_stats.update(batch["o"])
        if self.dimg != (0,):
            self.g_stats.update(batch["g"])

    def train(self, batch):
        feed_dict = {}
        feed_dict[self.demo_inputs_tf["o"]] = batch["o"]
        feed_dict[self.demo_inputs_tf["u"]] = batch["u"]
        if self.dimg != (0,):
            feed_dict[self.demo_inputs_tf["g"]] = batch["g"]
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss, _


class GANShaping(Shaping):
    def __init__(
        self,
        sess,
        dims,
        gamma,
        max_u,
        potential_weight,
        layer_sizes,
        latent_dim,
        gp_lambda,
        critic_iter,
        norm_obs,
        norm_eps,
        norm_clip,
    ):

        """
        GAN with Wasserstein distance plus gradient penalty.
        Args:
            gamma            (float) - discount factor
            demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
            potential_weight (float)
        """
        super().__init__(gamma)

        # Prepare parameters
        self.sess = sess
        self.dims = dims
        self.dimo = self.dims["o"]
        self.dimg = self.dims["g"]
        self.dimu = self.dims["u"]
        self.max_u = max_u
        self.layer_sizes = layer_sizes
        self.norm_obs = norm_obs
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.potential_weight = potential_weight
        self.critic_iter = critic_iter

        self.demo_inputs_tf = {}
        self.demo_inputs_tf["o"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimo))
        self.demo_inputs_tf["u"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimu))
        if self.dimg != (0,):
            self.demo_inputs_tf["g"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimg))

        # normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        demo_state_tf = self._concat_normalize_inputs(  # remove _normalize to not normalize the inputs
            self.demo_inputs_tf["o"],
            self.demo_inputs_tf["g"] if "g" in self.demo_inputs_tf.keys() else None,
            self.demo_inputs_tf["u"],
        )

        self.potential_weight = potential_weight
        self.critic_iter = critic_iter
        self.train_gen = 0  # counter

        # Generator & Discriminator
        self.generator = Generator(fc_layer_params=layer_sizes + [demo_state_tf.shape[-1]])
        self.discriminator = Discriminator(fc_layer_params=layer_sizes + [1])

        # Loss functions
        assert len(demo_state_tf.shape) >= 2
        fake_data = self.generator(tf.random.uniform([tf.shape(demo_state_tf)[0], latent_dim]))
        disc_fake = self.discriminator(fake_data)
        disc_real = self.discriminator(demo_state_tf)
        # discriminator loss on generator (including gp loss)
        alpha = tf.random.uniform(
            shape=[tf.shape(demo_state_tf)[0]] + [1] * (len(demo_state_tf.shape) - 1), minval=0.0, maxval=1.0
        )
        interpolates = alpha * demo_state_tf + (1.0 - alpha) * fake_data
        disc_interpolates = self.discriminator(interpolates)
        gradients = tf.gradients(disc_interpolates, interpolates)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
        self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) + gp_lambda * gradient_penalty
        # generator loss
        self.gen_cost = -tf.reduce_mean(disc_fake)

        # Train
        self.disc_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.disc_cost, var_list=self.discriminator.trainable_variables
        )
        self.disc_train_policy_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9
        ).minimize(self.disc_cost_policy, var_list=self.discriminator.trainable_variables)
        self.gen_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
            self.gen_cost, var_list=self.generator.trainable_variables
        )

        # Evaluation
        self.eval_generator = self.generator(tf.random.uniform([10, latent_dim]))

    def potential(self, o, g, u):
        """
        Use the output of the GAN's discriminator as potential.
        """
        state_tf = self._concat_normalize_inputs(o, g, u)  # remove _normalize to not normalize the inputs
        potential = self.discriminator(state_tf)
        potential = potential * self.potential_weight
        return potential

    def update_stats(self, batch):
        # add transitions to normalizer
        if not self.norm_obs:
            return
        self.o_stats.update(batch["o"])
        if self.dimg != (0,):
            self.g_stats.update(batch["g"])

    def train(self, batch):
        feed_dict = {}
        feed_dict[self.demo_inputs_tf["o"]] = batch["o"]
        feed_dict[self.demo_inputs_tf["u"]] = batch["u"]
        if self.dimg != (0,):
            feed_dict[self.demo_inputs_tf["g"]] = batch["g"]
        # train critic
        disc_cost, _ = sess.run([self.disc_cost, self.disc_train_op], feed_dict=feed_dict)
        # train generator
        if self.train_gen == 0:
            sess.run(self.gen_train_op, feed_dict=feed_dict)
        self.train_gen = np.mod(self.train_gen + 1, self.critic_iter)
        return disc_cost, _


# For Training
shaping_cls = {"nf": NFShaping, "gan": GANShaping}


class EnsembleRewardShapingWrapper:
    def __init__(self, num_ensembles, *args, **kwargs):
        self.shapings = [RewardShaping(*args, **kwargs) for _ in range(num_ensembles)]

    def train(self, *args, **kwargs):
        for i, shaping in enumerate(self.shapings):
            logger.log("Training shaping function #{}...".format(i))
            shaping.train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        for i, shaping in enumerate(self.shapings):
            logger.log("Evaluating shaping function #{}...".format(i))
            shaping.evaluate(*args, **kwargs)

    def potential(self, *args, **kwargs):
        potential = tf.reduce_mean([x.potential(*args, **kwargs) for x in self.shapings], axis=0)
        return potential

    def reward(self, *args, **kwargs):
        reward = tf.reduce_mean([x.reward(*args, **kwargs) for x in self.shapings], axis=0)
        return reward


class RewardShaping:
    def __init__(self,sess, dims, max_u, gamma, demo_strategy, num_epochs, batch_size, **shaping_params):

        if demo_strategy not in shaping_cls.keys():
            self.shaping = None
            return

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.shaping_params = shaping_params[demo_strategy].copy()

        # Update parameters
        self.shaping_params.update(
            {
                "sess": sess,
                "dims": dims,
                "max_u": max_u,
                "gamma": gamma,
                "norm_obs": shaping_params["norm_obs"],
                "norm_eps": shaping_params["norm_eps"],
                "norm_clip": shaping_params["norm_clip"],
            }
        )
        self.shaping = shaping_cls[demo_strategy](**self.shaping_params)


    def train(self, demo_data):
        self.shaping.update_stats(demo_data)

        for epoch in range(self.num_epochs):
            losses = np.empty(0)
            for (o, g, u) in iterbatches((demo_data["o"], demo_data["g"], demo_data["u"]), batch_size=self.batch_size):
                batch = {"o": o, "g": g, "u": u}
                d_loss, g_loss = self.shaping.train(batch)
                losses = np.append(losses, d_loss)
            if epoch % (self.num_epochs / 100) == (self.num_epochs / 100 - 1):
                logger.info("epoch: {} demo shaping loss: {}".format(epoch, np.mean(losses)))
                # print("epoch: {} demo shaping loss: {}".format(epoch, np.mean(losses)))
                mean_pot = self.shaping.evaluate(batch)
                logger.info("epoch: {} mean potential on demo data: {}".format(epoch, mean_pot))

        self.shaping.post_training_update(demo_data)

    def evaluate(self):
        return
        # input data - used for both training and test set
        dim1 = 0
        dim2 = 1
        num_point = 24
        ls = np.linspace(-1.0, 1.0, num_point)
        o_1, o_2 = np.meshgrid(ls, ls)
        o_r = 0.0 * np.ones((num_point ** 2, 4))  # TODO change this dimension
        o_r[..., dim1 : dim1 + 1] = o_1.reshape(-1, 1)
        o_r[..., dim2 : dim2 + 1] = o_2.reshape(-1, 1)
        u_r = 1.0 * np.ones((num_point ** 2, 2))  # TODO change this dimension

        o_tc = torch.tensor(o_r, dtype=torch.float).to(device)
        g_tc = torch.empty((o_tc.shape[0], 0)).to(device)
        u_tc = torch.tensor(u_r, dtype=torch.float).to(device)

        p = self.shaping.potential(o_tc, g_tc, u_tc).cpu().data.numpy()

        res = {"o": (o_1, o_2), "surf": p.reshape((num_point, num_point))}

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from td3fd.td3.debug import visualize_query

        plt.figure(0)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0], projection="3d")
        ax.clear()
        visualize_query.visualize_potential_surface(ax, res)
        plt.show(block=True)

    def potential(self, o, g, u):
        return self.shaping.potential(o, g, u)

    def reward(self, o, g, u, o_2, g_2, u_2):
        return self.shaping.reward(o, g, u, o_2, g_2, u_2)