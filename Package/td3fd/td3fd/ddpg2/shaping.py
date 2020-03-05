import itertools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from td3fd.ddpg2.normalizer import Normalizer
from td3fd import logger
from td3fd.memory import iterbatches
from td3fd.ddpg2.normalizing_flows import create_maf
from td3fd.ddpg2.gan_network import Generator, Discriminator


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


class NFShaping(Shaping):
    def __init__(
        self,
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
        self.init_args = locals()

        super(NFShaping, self).__init__(gamma)

        # Prepare parameters
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
        self.learning_rate = 2e-4
        self.scale = tf.constant(5.0, dtype=tf.float64)

        # normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip)

        # normalizing flow
        state_dim = self.dimo[0] + self.dimg[0] + self.dimu[0]
        self.nf = create_maf(dim=state_dim, num_bijectors=num_bijectors, layer_sizes=layer_sizes)
        # create weights
        self.nf.sample()
        # optimizers
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    @tf.function
    def potential(self, o, g, u):
        o = self.o_stats.normalize(o)
        g = self.g_stats.normalize(g)
        state_tf = tf.concat(axis=1, values=[o, g, u / self.max_u])

        state_tf = tf.cast(state_tf, tf.float64)

        potential = tf.reshape(self.nf.prob(state_tf), (-1, 1))
        potential = tf.math.log(potential + tf.exp(-self.scale))
        potential = potential + self.scale  # shift
        potential = self.potential_weight * potential / self.scale  # scale

        potential = tf.cast(potential, tf.float32)

        return potential

    def update_stats(self, batch):
        # add transitions to normalizer
        if not self.norm_obs:
            return
        self.o_stats.update(batch["o"])
        self.g_stats.update(batch["g"])

    def train(self, batch):

        o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
        g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
        u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)

        loss_tf = self.train_tf(o_tf, g_tf, u_tf)

        return loss_tf.numpy()

    def evaluate(self, batch):
        o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
        g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
        u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
        potential = self.potential(o_tf, g_tf, u_tf)
        potential = np.mean(potential.numpy())
        return potential

    @tf.function
    def train_tf(self, o_tf, g_tf, u_tf):
        o_tf = self.o_stats.normalize(o_tf)
        g_tf = self.g_stats.normalize(g_tf)
        state_tf = tf.concat(axis=1, values=[o_tf, g_tf, u_tf / self.max_u])

        state_tf = tf.cast(state_tf, tf.float64)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(state_tf)
                # loss function that tries to maximize log prob
                # log probability
                neg_log_prob = tf.clip_by_value(-self.nf.log_prob(state_tf), -1e5, 1e5)
                neg_log_prob = tf.reduce_mean(tf.reshape(neg_log_prob, (-1, 1)))
            # regularizer
            jacobian = tape2.gradient(neg_log_prob, state_tf)
            regularizer = tf.norm(jacobian, ord=2)
            loss_tf = self.prm_loss_weight * neg_log_prob + self.reg_loss_weight * regularizer
        grads = tape.gradient(loss_tf, self.nf.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nf.trainable_variables))

        loss_tf = tf.cast(loss_tf, tf.float32)

        return loss_tf

    def __getstate__(self):
        """
        Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        state = {k: v for k, v in self.init_args.items() if not k in ["self", "__class__"]}
        state["tf"] = {
            "o_stats": self.o_stats.get_weights(),
            "g_stats": self.g_stats.get_weights(),
            "nf": list(map(lambda v: v.numpy(), self.nf.variables)),
        }
        return state

    def __setstate__(self, state):
        stored_vars = state.pop("tf")
        self.__init__(**state)
        self.o_stats.set_weights(stored_vars["o_stats"])
        self.g_stats.set_weights(stored_vars["g_stats"])
        list(map(lambda v: v[0].assign(v[1]), zip(self.nf.variables, stored_vars["nf"])))


class GANShaping(Shaping):
    def __init__(
        self,
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
        self.init_args = locals()

        super(GANShaping, self).__init__(gamma)

        # Prepare parameters
        self.dims = dims
        self.dimo = self.dims["o"]
        self.dimg = self.dims["g"]
        self.dimu = self.dims["u"]
        self.max_u = max_u
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.norm_obs = norm_obs
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.potential_weight = potential_weight
        self.critic_iter = critic_iter
        self.gp_lambda = gp_lambda

        # normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip)

        # Generator & Discriminator
        state_dim = self.dimo[0] + self.dimg[0] + self.dimu[0]
        self.generator = Generator(layer_sizes=layer_sizes + [state_dim])
        self.discriminator = Discriminator(layer_sizes=layer_sizes + [1])
        # create weights
        self.generator(tf.zeros([0, latent_dim]))
        self.discriminator(tf.zeros([0, state_dim]))

        # Train
        self.train_gen = tf.Variable(0, trainable=False)  # counter
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

    @tf.function
    def potential(self, o, g, u):
        """
        Use the output of the GAN's discriminator as potential.
        """
        o = self.o_stats.normalize(o)
        g = self.g_stats.normalize(g)
        state_tf = tf.concat(axis=1, values=[o, g, u / self.max_u])

        potential = self.discriminator(state_tf)
        potential = self.potential_weight * potential
        return potential

    def update_stats(self, batch):
        # add transitions to normalizer
        if not self.norm_obs:
            return
        self.o_stats.update(batch["o"])
        self.g_stats.update(batch["g"])

    def train(self, batch):

        o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
        g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
        u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)

        disc_loss_tf = self.train_tf(o_tf, g_tf, u_tf)

        return disc_loss_tf.numpy()

    def evaluate(self, batch):
        o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
        g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
        u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
        potential = self.potential(o_tf, g_tf, u_tf)
        potential = np.mean(potential.numpy())
        return potential

    @tf.function
    def train_tf(self, o_tf, g_tf, u_tf):

        o_tf = self.o_stats.normalize(o_tf)
        g_tf = self.g_stats.normalize(g_tf)
        state_tf = tf.concat(axis=1, values=[o_tf, g_tf, u_tf / self.max_u])

        with tf.GradientTape(persistent=True) as tape:
            fake_data = self.generator(tf.random.uniform([tf.shape(state_tf)[0], self.latent_dim]))
            disc_fake = self.discriminator(fake_data)
            disc_real = self.discriminator(state_tf)
            # discriminator loss on generator (including gp loss)
            alpha = tf.random.uniform(
                shape=[tf.shape(state_tf)[0]] + [1] * (len(tf.shape(state_tf)) - 1), minval=0.0, maxval=1.0
            )
            interpolates = alpha * state_tf + (1.0 - alpha) * fake_data
            with tf.GradientTape() as tape2:
                tape2.watch(interpolates)
                disc_interpolates = self.discriminator(interpolates)
            gradients = tape2.gradient(disc_interpolates, interpolates)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
            disc_loss_tf = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) + self.gp_lambda * gradient_penalty
            # generator loss
            gen_loss_tf = -tf.reduce_mean(disc_fake)
        disc_grads = tape.gradient(disc_loss_tf, self.discriminator.trainable_weights)
        gen_grads = tape.gradient(gen_loss_tf, self.generator.trainable_weights)

        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))
        if self.train_gen % self.critic_iter == 0:
            self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))
        self.train_gen.assign_add(1)

        return disc_loss_tf

    def __getstate__(self):
        """
        Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        state = {k: v for k, v in self.init_args.items() if not k in ["self", "__class__"]}
        state["tf"] = {
            "o_stats": self.o_stats.get_weights(),
            "g_stats": self.g_stats.get_weights(),
            "discriminator": self.discriminator.get_weights(),
            "generator": self.generator.get_weights(),
        }
        return state

    def __setstate__(self, state):
        stored_vars = state.pop("tf")
        self.__init__(**state)
        self.o_stats.set_weights(stored_vars["o_stats"])
        self.g_stats.set_weights(stored_vars["g_stats"])
        self.discriminator.set_weights(stored_vars["discriminator"])
        self.generator.set_weights(stored_vars["generator"])


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

    @tf.function
    def potential(self, o, g, u):
        potential = tf.reduce_mean([x.potential(o, g, u) for x in self.shapings], axis=0)
        return potential

    @tf.function
    def reward(self, o, g, u, o_2, g_2, u_2):
        reward = tf.reduce_mean([x.reward(o, g, u, o_2, g_2, u_2) for x in self.shapings], axis=0)
        return reward


class RewardShaping:
    def __init__(self, dims, max_u, gamma, demo_strategy, num_epochs, batch_size, **shaping_params):

        if demo_strategy not in shaping_cls.keys():
            self.shaping = None
            return

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.shaping_params = shaping_params[demo_strategy].copy()

        # Update parameters
        self.shaping_params.update(
            {
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
                loss = self.shaping.train(batch)
                losses = np.append(losses, loss)
            if epoch % (self.num_epochs / 100) == (self.num_epochs / 100 - 1):
                logger.info("epoch: {} demo shaping loss: {}".format(epoch, np.mean(losses)))
                mean_pot = self.shaping.evaluate(batch)
                logger.info("epoch: {} mean potential on demo data: {}".format(epoch, mean_pot))

        self.shaping.post_training_update(demo_data)

    def evaluate(self, demo_data):
        return

        import matplotlib.pyplot as plt

        x = []
        y = []
        for var in np.arange(0, 1e-2, 1e-4):
            o_tf = tf.convert_to_tensor(
                demo_data["o"] + np.random.normal(0.0, var, demo_data["o"].shape), dtype=tf.float32
            )
            g_tf = tf.convert_to_tensor(
                demo_data["g"] + np.random.normal(0.0, var, demo_data["g"].shape), dtype=tf.float32
            )
            u_tf = tf.convert_to_tensor(
                demo_data["u"] + np.random.normal(0.0, var, demo_data["u"].shape), dtype=tf.float32
            )
            p_tf = self.shaping.potential(o=o_tf, g=g_tf, u=u_tf)
            x.append(var)
            y.append(np.mean(p_tf.numpy()))

        plt.plot(x, y)
        plt.xlabel("var")
        plt.ylabel("potential")
        plt.savefig("evaluate_potential.png", dpi=200)
        plt.show()

    @tf.function
    def potential(self, o, g, u):
        return self.shaping.potential(o, g, u)

    @tf.function
    def reward(self, o, g, u, o_2, g_2, u_2):
        return self.shaping.reward(o, g, u, o_2, g_2, u_2)
