import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torchsummary import summary
# from torchvision import utils

import td3fd.td3.normalizing_flow as fnn
from td3fd import logger
from td3fd.memory import iterbatches
from td3fd.td3.gan_network import Discriminator, Generator
from td3fd.td3.gan_network_img import Discriminator as DiscriminatorImg
from td3fd.td3.gan_network_img import Generator as GeneratorImg
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

    def train(self, batch):
        pass

    def evaluate(self, batch):
        pass

    def post_training_update(self, batch):
        """
        """
        pass


# debug shaping: only for reacher 2d environment
# class DbgShaping(Shaping):
#     def potential(self, o, g, u):
#         return -20 * torch.norm(o-g, dim=1, keepdim=True)


class NFShaping(Shaping):
    def __init__(
        self,
        dims,
        max_u,
        gamma,
        num_blocks,
        num_hidden,
        potential_weight,
        norm_obs,
        norm_eps,
        norm_clip,
        prm_loss_weight,
        reg_loss_weight,
        **kwargs
    ):
        # Store initial args passed into the function
        self.init_args = locals()

        super().__init__(gamma)

        # Prepare parameters
        self.dims = dims
        self.dimo = self.dims["o"]
        self.dimg = self.dims["g"]
        self.dimu = self.dims["u"]
        self.max_u = max_u
        self.num_blocks = num_blocks
        self.num_hidden = num_hidden
        self.norm_obs = norm_obs
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.prm_loss_weight = prm_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.potential_weight = potential_weight

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.scale = torch.tensor(5.0).to(device)

        # Normalizer for goal and clipervation.
        self.o_stats = Normalizer(self.dimo, norm_eps, norm_clip).to(device)
        self.g_stats = Normalizer(self.dimg, norm_eps, norm_clip).to(device)

        # Normalizing flow
        num_inputs = self.dimo[0] + self.dimu[0] + self.dimg[0]
        modules = []
        for _ in range(self.num_blocks):
            modules += [
                fnn.MADE(num_inputs, self.num_hidden),
                # fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs),
            ]
        self.model = fnn.FlowSequential(*modules)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.fill_(0)
        self.model.to(device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-6)

    def update_stats(self, batch):
        # add transitions to normalizer
        if not self.norm_obs:
            return
        self.o_stats.update(torch.tensor(batch["o"], dtype=torch.float).to(device))
        self.g_stats.update(torch.tensor(batch["g"], dtype=torch.float).to(device))

    def train(self, batch):

        o_tc = torch.tensor(batch["o"], dtype=torch.float).to(device)
        g_tc = torch.tensor(batch["g"], dtype=torch.float).to(device)
        u_tc = torch.tensor(batch["u"], dtype=torch.float).to(device)

        # Do not normalize input o/g for gym/mujoco envs
        if self.norm_obs:
            o_tc = self.o_stats.normalize(o_tc)
            g_tc = self.g_stats.normalize(g_tc)
        u_tc = u_tc / self.max_u

        inputs = torch.cat((o_tc, g_tc, u_tc), axis=1)

        prm_loss = -self.model.log_probs(inputs)
        prm_loss = prm_loss.mean()
        # calculate gradients of log prob with respect to inputs
        inputs_grad = inputs.detach().requires_grad_()
        log_prob = self.model.log_probs(inputs_grad)
        assert not any(torch.isnan(log_prob)), "Found NaN in log_prob {}.".format(log_prob)
        gradients = autograd.grad(
            outputs=log_prob,
            inputs=inputs_grad,
            grad_outputs=torch.ones_like(log_prob),
            create_graph=True,
            retain_graph=True,
        )[0]
        reg_loss = gradients.norm(2, dim=1).mean()
        loss = reg_loss * self.reg_loss_weight + prm_loss * self.prm_loss_weight
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, 0

    def potential(self, o, g, u):
        """
        Use the output of the GAN's discriminator as potential.
        """
        # Do not normalize input o/g for gym/mujoco envs
        if self.norm_obs:
            o = self.o_stats.normalize(o)
            g = self.g_stats.normalize(g)
        u = u / self.max_u

        inputs = torch.cat((o, g, u), axis=1)
        potential = self.model.log_probs(inputs)
        # clip the potential and shift it to [0, inf]
        potential = torch.log(torch.exp(potential) + torch.exp(-self.scale))
        potential = potential / self.scale + 1
        # treat NaN as 0
        potential = torch.where(torch.isnan(potential), torch.zeros_like(potential), potential)
        # scale up
        potential = potential * self.potential_weight

        return potential

    def evaluate(self, batch):
        o = torch.tensor(batch["o"], dtype=torch.float).to(device)
        g = torch.tensor(batch["g"], dtype=torch.float).to(device)
        u = torch.tensor(batch["u"], dtype=torch.float).to(device)
        return self.potential(o, g, u).mean().cpu().data

    def __getstate__(self):
        """
        Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        state = {k: v for k, v in self.init_args.items() if not k == "self"}
        state["tc"] = {
            "o_stats": self.o_stats.state_dict(),
            "g_stats": self.g_stats.state_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return state

    def __setstate__(self, state):
        state_dicts = state.pop("tc")
        self.__init__(**state)
        self.o_stats.load_state_dict(state_dicts["o_stats"])
        self.g_stats.load_state_dict(state_dicts["g_stats"])
        self.model.load_state_dict(state_dicts["model"])
        self.optimizer.load_state_dict(state_dicts["optimizer"])


class GANShaping(Shaping):
    def __init__(
        self,
        dims,
        max_u,
        gamma,
        layer_sizes,
        potential_weight,
        norm_obs,
        norm_eps,
        norm_clip,
        latent_dim,
        lambda_term,
        gp_target,
        sub_potential_mean,
        **kwargs
    ):

        # Store initial args passed into the function
        self.init_args = locals()

        super().__init__(gamma)

        # Prepare parameters
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

        # WGAN values from paper
        self.latent_dim = latent_dim  # 100 for images, use a smaller value for state based environments
        # lambda set to 10 for images, but we use a smaller value for faster convergence when output dim is low
        self.lambda_term = lambda_term
        self.gp_target = gp_target
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.critic_iter = 5
        self.batch_size = 64

        # Normalizer for goal and clipervation.
        self.o_stats = Normalizer(self.dimo, norm_eps, norm_clip).to(device)
        self.g_stats = Normalizer(self.dimg, norm_eps, norm_clip).to(device)

        state_dim = self.dimo[0] + self.dimg[0] + self.dimu[0]
        self.G = Generator(self.latent_dim, state_dim, self.layer_sizes).to(device)
        summary(self.G, (self.latent_dim,))
        self.D = Discriminator(state_dim, self.layer_sizes).to(device)
        summary(self.D, (state_dim,))
        self.C = None  # number of channels

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.current_critic_iter = 0

        # add running mean output
        self.sub_potential_mean = sub_potential_mean
        self.potential_mean = 0.0

    def update_stats(self, batch):
        # add transitions to normalizer
        if not self.norm_obs:
            return
        self.o_stats.update(torch.tensor(batch["o"], dtype=torch.float).to(device))
        self.g_stats.update(torch.tensor(batch["g"], dtype=torch.float).to(device))

    def post_training_update(self, batch):
        o = torch.tensor(batch["o"], dtype=torch.float).to(device)
        g = torch.tensor(batch["g"], dtype=torch.float).to(device)
        u = torch.tensor(batch["u"], dtype=torch.float).to(device)
        if self.sub_potential_mean:
            self.potential_mean = self.potential(o, g, u).mean().cpu().data

    def train(self, batch):

        o_tc = torch.tensor(batch["o"], dtype=torch.float).to(device)
        g_tc = torch.tensor(batch["g"], dtype=torch.float).to(device)
        u_tc = torch.tensor(batch["u"], dtype=torch.float).to(device)

        # Do not normalize input o/g for gym/mujoco envs
        if self.norm_obs:
            o_tc = self.o_stats.normalize(o_tc)
            g_tc = self.g_stats.normalize(g_tc)
        u_tc = u_tc / self.max_u

        images = torch.cat((o_tc, g_tc, u_tc), axis=1)

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
        eta = (
            torch.FloatTensor(real_images_grad.shape[0], *([1] * (len(real_images_grad.shape) - 1)))
            .uniform_(0, 1)
            .to(device)
        )
        eta = eta.expand(*real_images_grad.shape)
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
        grad_penalty = ((gradients.norm(2, dim=1) - self.gp_target) ** 2).mean() * self.lambda_term
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
        # Do not normalize input o/g for gym/mujoco envs
        if self.norm_obs:
            o = self.o_stats.normalize(o)
            g = self.g_stats.normalize(g)
        u = u / self.max_u

        inputs = torch.cat((o, g, u), axis=1)
        potential = self.D(inputs)
        potential = potential - self.potential_mean
        potential = potential * self.potential_weight
        return potential

    def evaluate(self, batch):
        o = torch.tensor(batch["o"], dtype=torch.float).to(device)
        g = torch.tensor(batch["g"], dtype=torch.float).to(device)
        u = torch.tensor(batch["u"], dtype=torch.float).to(device)
        return self.potential(o, g, u).mean().cpu().data

    def __getstate__(self):
        """
        Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        state = {k: v for k, v in self.init_args.items() if not k == "self"}
        state["tc"] = {
            "o_stats": self.o_stats.state_dict(),
            "g_stats": self.g_stats.state_dict(),
            "D": self.D.state_dict(),
            "G": self.G.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "potential_mean": self.potential_mean,
        }
        return state

    def __setstate__(self, state):
        state_dicts = state.pop("tc")
        self.__init__(**state)
        self.o_stats.load_state_dict(state_dicts["o_stats"])
        self.g_stats.load_state_dict(state_dicts["g_stats"])
        self.D.load_state_dict(state_dicts["D"])
        self.G.load_state_dict(state_dicts["G"])
        self.d_optimizer.load_state_dict(state_dicts["d_optimizer"])
        self.g_optimizer.load_state_dict(state_dicts["g_optimizer"])
        self.potential_mean = state_dicts["potential_mean"]


class ImgGANShaping(Shaping):
    def __init__(self, dims, max_u, gamma, layer_sizes, potential_weight, norm_obs, norm_eps, norm_clip, **kwargs):

        # Store initial args passed into the function
        self.init_args = locals()

        super().__init__(gamma)

        # Prepare parameters
        self.dims = dims
        self.dimo = self.dims["o"]
        self.dimg = self.dims["g"]
        self.dimu = self.dims["u"]
        self.max_u = max_u
        self.potential_weight = potential_weight

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.latent_dim = 100
        self.b1 = 0.5
        self.b2 = 0.999
        self.critic_iter = 5
        self.lambda_term = 10
        self.batch_size = 64
        self.channel = 3  # use 4 when using RGBD input

        # Normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, norm_eps, norm_clip).to(device)
        self.g_stats = Normalizer(self.dimg, norm_eps, norm_clip).to(device)

        # state_dim = self.dimo[0] + self.dimg[0] + self.dimu[0]
        state_dim = (self.channel, 32, 32)
        self.G = GeneratorImg(self.latent_dim, self.channel).to(device)
        summary(self.G, (self.latent_dim, 1, 1))
        self.D = DiscriminatorImg(self.channel).to(device)
        summary(self.D, state_dim)
        self.C = None  # number of channels

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.current_critic_iter = 0

    def train(self, batch):

        o_tc = torch.FloatTensor(batch["o"]).to(device)
        g_tc = torch.FloatTensor(batch["g"]).to(device)
        u_tc = torch.FloatTensor(batch["u"]).to(device)

        u_tc = u_tc / self.max_u

        # normalize the images with depth
        # images[:, :3, ...] = images[:, :3, ...].div(255.0 / 2).add(-1.0)
        # images[:, 3:4, ...] = (images[:, 3:4, ...] - images[:, 3:4, ...].mean()) / images[:, 3:4, ...].var()
        # normalize the image without depth

        # TODO:
        # assert False, "here we should find a proper way of concatenate observations and actions"
        # normalize the images without depth
        images = o_tc[:, :3, ...].div(255.0 / 2).add(-1.0)

        # Train discriminator
        # requires grad, Generator requires_grad = False
        self.D.requires_grad_(True)
        z = torch.randn((images.shape[0],) + (self.latent_dim, 1, 1)).to(device)
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
        eta = (
            torch.FloatTensor(real_images_grad.shape[0], *([1] * (len(real_images_grad.shape) - 1)))
            .uniform_(0, 1)
            .to(device)
        )
        eta = eta.expand(*real_images_grad.shape)
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
        W_distance = -d_loss_fake - d_loss_real

        self.D.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        g_loss = 0.0
        if self.current_critic_iter % self.critic_iter == 0:
            # Train generator
            self.D.requires_grad_(False)
            # compute loss with fake images
            z = torch.randn((images.shape[0],) + (self.latent_dim, 1, 1)).to(device)
            fake_images = self.G(z)
            g_loss = -self.D(fake_images)
            g_loss = g_loss.mean()

            self.G.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

        self.current_critic_iter += 1
        return W_distance, g_loss

    def potential(self, o, g, u):
        """
        Use the output of the GAN's discriminator as potential.
        """
        u = u / self.max_u
        assert False, "need to find a proper way to concatenate obs, goal and action. depends what you do in training"
        inputs = torch.cat((o, g, u), axis=1)
        potential = self.D(inputs)
        potential = potential * self.potential_weight
        return potential

    def evaluate(self):
        # TODO: fix utils imports on cluster
        # n = 16  # generate 16 images
        # z = torch.randn(n, self.latent_dim, 1, 1).to(device)  # latent vector
        # samples = self.G(z)  # pass to generator
        # samples = samples.mul(0.5).add(0.5)  # change output from (-1, 1) to (0, 1)
        # samples = samples.data.cpu()
        # grid = utils.make_grid(samples[:, :3, ...])
        # print("Grid of 2x8 images saved to 'dgan_model_image.png'.")
        # utils.save_image(grid, "dgan_model_image.png")

    def real_images(self, images, number_of_images):
        if self.C == 3:
            return images.view(-1, self.C, 32, 32)[: self.number_of_images].data.cpu().numpy()
        else:
            return images.view(-1, 32, 32)[: self.number_of_images].data.cpu().numpy()

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def __getstate__(self):
        """
        Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        state = {k: v for k, v in self.init_args.items() if not k == "self"}
        state["tc"] = {
            "o_stats": self.o_stats.state_dict(),
            "g_stats": self.g_stats.state_dict(),
            "D": self.D.state_dict(),
            "G": self.G.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
        }
        return state

    def __setstate__(self, state):
        state_dicts = state.pop("tc")
        self.__init__(**state)
        self.o_stats.load_state_dict(state_dicts["o_stats"])
        self.g_stats.load_state_dict(state_dicts["g_stats"])
        self.D.load_state_dict(state_dicts["D"])
        self.G.load_state_dict(state_dicts["G"])
        self.d_optimizer.load_state_dict(state_dicts["d_optimizer"])
        self.g_optimizer.load_state_dict(state_dicts["g_optimizer"])


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
        potentials = [x.potential(*args, **kwargs) for x in self.shapings]
        return torch.mean(torch.stack(potentials), dim=0)

    def reward(self, *args, **kwargs):
        rewards = [x.reward(*args, **kwargs) for x in self.shapings]
        return torch.mean(torch.stack(rewards), dim=0)

    def __getstate__(self):
        """
        We only save the shaping class. after reloading, only potential and reward functions can be used.
        """
        state = {"shaping": self.shapings}
        return state

    def __setstate__(self, state):
        self.shapings = state["shaping"]


class RewardShaping:
    def __init__(self, env, demo_strategy, discount, num_epochs, batch_size, **shaping_params):

        if demo_strategy not in shaping_cls.keys():
            self.shaping = None
            return

        self.shaping_params = shaping_params[demo_strategy]

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        dims = {
            "o": env.observation_space["observation"].shape,  # for td3fd
            "g": env.observation_space["desired_goal"].shape,
            # "o": env.observation_space.shape,  # for rlkit
            # "g": (0,),
            #
            "u": env.action_space.shape,
        }
        max_u = env.action_space.high[0]
        # assert max_u == 1.0  # for rlkit
        # Update parameters
        self.shaping_params.update(
            {
                "dims": dims,  # agent takes an input observations
                "max_u": max_u,
                "gamma": discount,
                "norm_obs": shaping_params["norm_obs"],
                "norm_eps": shaping_params["norm_eps"],
                "norm_clip": shaping_params["norm_clip"],
            }
        )
        # print(self.shaping_params)
        self.shaping = shaping_cls[demo_strategy](**self.shaping_params)

    def _convert_rlkit_to_td3fd(self, demo_data):
        converted_demo_data = dict()
        keys = demo_data[0].keys()
        for path in demo_data:
            for key in keys:
                if key in converted_demo_data.keys():
                    if type(path[key]) == list:
                        converted_demo_data[key] += path[key]
                    else:
                        converted_demo_data[key] = np.concatenate((converted_demo_data[key], path[key]), axis=0)
                else:
                    converted_demo_data[key] = path[key]
        demo_data = converted_demo_data

        demo_data["o"] = demo_data["observations"]
        demo_data["u"] = demo_data["actions"]
        assert len(demo_data["o"].shape) == 2
        demo_data["g"] = np.empty((demo_data["o"].shape[0], 0))
        return demo_data

    def train(self, demo_data):
        # for rlkit
        # demo_data = self._convert_rlkit_to_td3fd(demo_data)
        #
        self.shaping.update_stats(demo_data)

        for epoch in range(self.num_epochs):
            losses = np.empty(0)
            for (o, g, u) in iterbatches((demo_data["o"], demo_data["g"], demo_data["u"]), batch_size=self.batch_size):
                batch = {"o": o, "g": g, "u": u}
                d_loss, g_loss = self.shaping.train(batch)
                losses = np.append(losses, d_loss.cpu().data.numpy())
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
        # for rlkit
        # assert len(o.shape) == 2
        # g = torch.empty((o.shape[0], 0)).to(device)
        #
        potential = self.shaping.potential(o, g, u)
        assert not any(torch.isnan(potential)), "Found NaN in potential {}.".format(potential)
        return potential

    def reward(self, o, g, u, o_2, g_2, u_2):
        # for rlkit
        # assert len(o.shape) == 2
        # g = torch.empty((o.shape[0], 0)).to(device)
        # g_2 = torch.empty((o_2.shape[0], 0)).to(device)
        #
        return self.shaping.reward(o, g, u, o_2, g_2, u_2)

    def __getstate__(self):
        """
        We only save the shaping class. after reloading, only potential and reward functions can be used.
        """
        state = {"shaping": self.shaping}
        return state

    def __setstate__(self, state):
        self.shaping = state["shaping"]
