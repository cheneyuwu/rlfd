import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from torchsummary import summary
from td3fd.td3.gan_network import Generator, Discriminator
from td3fd.td3.normalizer import Normalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Shaping:
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

    def potential(self, o, g, u):
        raise NotImplementedError

    def reward(self, o, g, u, o_2, g_2, u_2):
        potential = self.potential(o, g, u)
        next_potential = self.potential(o_2, g_2, u_2)
        assert potential.shape[1] == next_potential.shape[1] == 1
        return self.gamma * next_potential - potential


class GANShaping(Shaping):
    def __init__(self, dims, max_u, gamma, layer_sizes, potential_weight, **kwargs):

        super().__init__(gamma)

        # Prepare parameters
        self.dims = dims
        self.dimo = self.dims["o"]
        self.dimg = self.dims["g"]
        self.dimu = self.dims["u"]
        self.max_u = max_u

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.latent_dim = 2
        self.b1 = 0.5
        self.b2 = 0.999
        self.critic_iter = 5
        self.lambda_term = 0.1  # used to be 10
        self.batch_size = 64

        # Normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo).to(device)
        self.g_stats = Normalizer(self.dimg).to(device)

        state_dim = self.dimo[0] + self.dimg[0] + self.dimu[0]
        self.G = Generator(self.latent_dim, state_dim).to(device)
        summary(self.G, (self.latent_dim,))
        self.D = Discriminator(state_dim).to(device)
        summary(self.D, (state_dim,))
        self.C = None  # number of channels

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.current_critic_iter = 0

    def train(self, batch):

        o_tc = self.o_stats.normalize(torch.FloatTensor(batch["o"]).to(device))
        g_tc = self.g_stats.normalize(torch.FloatTensor(batch["g"]).to(device))
        u_tc = torch.FloatTensor(batch["u"]).to(device)

        images = torch.cat((o_tc, g_tc, u_tc / self.max_u), axis=1)

        # Train discriminator
        # requires grad, Generator requires_grad = False
        self.D.requires_grad_(True)
        z = torch.randn((images.shape[0],) + (self.latent_dim,)).to(device)
        # train with real images
        d_loss_real = -self.D(images)
        d_loss_real = d_loss_real.mean()
        # train with fake images
        fake_images = self.G(z)
        d_loss_fake = self.D(fake_images)
        d_loss_fake = d_loss_fake.mean()
        # train with gradient penalty
        real_images_grad = images.detach()
        fake_images_grad = fake_images.detach()
        eta = torch.empty_like(real_images_grad).uniform_(0, 1).to(device)
        interpolated = eta * real_images_grad + ((1 - eta) * fake_images_grad)
        interpolated.requires_grad_()
        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)
        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        d_loss = d_loss_fake + d_loss_real + grad_penalty

        self.D.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        g_loss = 0.0
        if self.current_critic_iter % self.critic_iter == 0:
            # Train generator
            self.D.requires_grad_(False)
            # compute loss with fake images
            z = torch.randn((images.shape[0],) + (self.latent_dim,)).to(device)
            fake_images = self.G(z)
            g_loss = -self.D(fake_images)
            g_loss = g_loss.mean()

            self.G.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

        self.current_critic_iter += 1
        return d_loss, g_loss

    def potential(self, o, g, u):
        """
        Use the output of the GAN's discriminator as potential.
        """
        inputs = torch.cat((o, g, u / self.max_u), axis=1)
        potential = self.D(inputs)
        potential = potential * self.potential_weight
        return potential

    # def train_wrapper(self, train_loader):
    #     self.generator_iters = 1000
    #     # Now batches are callable self.data.next()
    #     self.data = self.get_infinite_batches(train_loader)
    #     for _ in range(self.generator_iters):
    #         self.train(self.data.__next__().to(device))

    def evaluate(self):
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, "dgan_model_image.png")

    def real_images(self, images, number_of_images):
        if self.C == 3:
            return self.to_np(images.view(-1, self.C, 32, 32)[: self.number_of_images]).data.cpu().numpy()
        else:
            return self.to_np(images.view(-1, 32, 32)[: self.number_of_images]).data.cpu().numpy()

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images


# class GANDemoShaping(DemoShaping):
#     def __init__(
#         self,
#         demo_inputs_tf,
#         policy_inputs_tf,
#         gamma,
#         o_stats,
#         g_stats,
#         potential_weight,
#         layer_sizes,
#         latent_dim,
#         gp_lambda,
#         critic_iter,
#         **kwargs,
#     ):
#         pass

#     def potential(self, o, g, u):
#         """
#         Use the output of the GAN's discriminator as potential.
#         """
#         state_tf = self._concat_normalize_inputs(o, g, u)  # remove _normalize to not normalize the inputs
#         # TODO: for pixel inputs
#         # state_tf = o
#         potential = self.discriminator(state_tf)
#         potential = potential * self.potential_weight
#         return potential

# def train(self, sess, feed_dict={}):
#     # train critic
#     disc_cost, _ = sess.run([self.disc_cost, self.disc_train_op], feed_dict=feed_dict)
#     # train generator
#     if self.train_gen == 0:
#         sess.run(self.gen_train_op, feed_dict=feed_dict)
#     self.train_gen = np.mod(self.train_gen + 1, self.critic_iter)
#     return disc_cost

# def train_on_policy(self, sess, feed_dict={}):
#     # train critic
#     disc_cost, _ = sess.run([self.disc_cost_policy, self.disc_train_policy_op], feed_dict=feed_dict)
#     return disc_cost

# def evaluate(self, sess, feed_dict={}):
#     pass


# class EnsGANDemoShaping(DemoShaping):
#     def __init__(
#         self,
#         demo_inputs_tf,
#         policy_inputs_tf,
#         num_ens,
#         gamma,
#         o_stats,
#         g_stats,
#         layer_sizes,
#         latent_dim,
#         gp_lambda,
#         critic_iter,
#         potential_weight,
#     ):
#         """
#         Args:
#             num_ens          (int)   - number of ensembles
#             gamma            (float) - discount factor
#             demo_inputs_tf           - demo_inputs that contains all the transitons from demonstration
#             potential_weight (float)
#         """
#         # Parameters for training
#         self.critic_iter = critic_iter
#         self.train_gen = 0  # counter
#         # Setup ensemble
#         self.gans = []
#         for i in range(num_ens):
#             self.gans.append(
#                 GANDemoShaping(
#                     demo_inputs_tf=demo_inputs_tf,
#                     policy_inputs_tf=policy_inputs_tf,
#                     gamma=gamma,
#                     o_stats=o_stats,
#                     g_stats=g_stats,
#                     layer_sizes=layer_sizes,
#                     latent_dim=latent_dim,
#                     gp_lambda=gp_lambda,
#                     critic_iter=critic_iter,
#                     potential_weight=potential_weight,
#                 )
#             )

#         # Loss functions
#         self.disc_cost = tf.reduce_sum([ens.disc_cost for ens in self.gans], axis=0)
#         self.disc_cost_policy = tf.reduce_sum([ens.disc_cost_policy for ens in self.gans], axis=0)
#         self.gen_cost = tf.reduce_sum([ens.gen_cost for ens in self.gans], axis=0)

#         # Training
#         self.disc_vars = list(itertools.chain(*[ens.discriminator.trainable_variables for ens in self.gans]))
#         self.gen_vars = list(itertools.chain(*[ens.generator.trainable_variables for ens in self.gans]))
#         self.disc_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
#             self.disc_cost, var_list=self.disc_vars
#         )
#         self.disc_train_policy_op = tf.compat.v1.train.AdamOptimizer(
#             learning_rate=1e-4, beta1=0.5, beta2=0.9
#         ).minimize(self.disc_cost_policy, var_list=self.disc_vars)
#         self.gen_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
#             self.gen_cost, var_list=self.gen_vars
#         )

#         super().__init__(gamma)

#     def potential(self, o, g, u):
#         # return the mean potential of all ens
#         potential = tf.reduce_mean([ens.potential(o=o, g=g, u=u) for ens in self.gans], axis=0)
#         assert potential.shape[1] == 1
#         return potential

#     def train(self, sess, feed_dict={}):
#         # train critic
#         disc_cost, _ = sess.run([self.disc_cost, self.disc_train_op], feed_dict=feed_dict)
#         # train generator
#         if self.train_gen == 0:
#             sess.run(self.gen_train_op, feed_dict=feed_dict)
#         self.train_gen = np.mod(self.train_gen + 1, self.critic_iter)
#         return disc_cost

#     def train_on_policy(self, sess, feed_dict={}):
#         # train critic
#         disc_cost, _ = sess.run([self.disc_cost_policy, self.disc_train_policy_op], feed_dict=feed_dict)
#         return disc_cost
