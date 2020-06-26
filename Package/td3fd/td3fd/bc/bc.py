import pickle

import numpy as np
import torch
import torch.nn.functional as F

# TODO switch between networks for image input and state input. image assumed to be (3, 32, 32)
from td3fd.bc.actorcritic_network import Actor, Critic

# from td3fd.td3.actorcritic_network_img import Actor, Critic # this is for image

from td3fd.td3.normalizer import Normalizer
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BC(object):
    def __init__(
        self,
        # for learning
        num_epochs,
        batch_size,
        # configuration
        dims,
        layer_sizes,
        pi_lr,
        max_u,
        norm_obs,
        norm_eps,
        norm_clip,
        info,
    ):
        # Store initial args passed into the function
        self.init_args = locals()

        # Parameters
        self.dims = dims
        self.layer_sizes = layer_sizes
        self.pi_lr = pi_lr
        self.max_u = max_u

        self.norm_obs = norm_obs
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip

        self.info = info

        # Prepare parameters
        self.dimo = self.dims["o"]
        self.dimg = self.dims["g"]
        self.dimu = self.dims["u"]

        # Normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip).to(device)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip).to(device)

        # Models
        # policy
        self.policy = Actor(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.pi_lr)
        summary(self.policy, [self.dimo, self.dimg])

    def get_actions(self, o, g, *args, **kwargs):
        o = o.reshape((-1, *self.dimo))
        g = g.reshape((o.shape[0], *self.dimg))
        o_tc = torch.tensor(o, dtype=torch.float).to(device)
        g_tc = torch.tensor(g, dtype=torch.float).to(device)

        # Normalize inputs (do not normalize inputs for gym/mujoco envs)
        if self.norm_obs:
            o_tc = self.o_stats.normalize(o_tc)
            g_tc = self.g_stats.normalize(g_tc)

        # TODO: uncomment for images
        # o_tc = (o_tc * 2.0 / 255.0) - 1.0

        u_tc = self.policy(o=o_tc, g=g_tc)

        u = u_tc.cpu().data.numpy()
        if o.shape[0] == 1:
            u = u[0]

        return u

    def save(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def train(self, batch, demo_batch=None):

        o_tc = torch.tensor(batch["o"], dtype=torch.float).to(device)
        g_tc = torch.tensor(batch["g"], dtype=torch.float).to(device)
        u_tc = torch.tensor(batch["u"], dtype=torch.float).to(device)

        # Normalize states and actions
        if self.norm_obs:
            o_tc = self.o_stats.normalize(o_tc)
            g_tc = self.g_stats.normalize(g_tc)

        # TODO: uncomment for images
        # o_tc = (o_tc * 2.0 / 255.0) - 1.0
        # o_2_tc = (o_2_tc * 2.0 / 255.0) - 1.0

        # Policy update
        loss = torch.mean((self.policy(o=o_tc, g=g_tc) - u_tc) ** 2)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss

    def update_stats(self, batch):
        # add transitions to normalizer
        if not self.norm_obs:
            return
        self.o_stats.update(torch.tensor(batch["o"], dtype=torch.float).to(device))
        self.g_stats.update(torch.tensor(batch["g"], dtype=torch.float).to(device))

    def logs(self, prefix=""):
        logs = []
        logs.append((prefix + "stats_o/mean", self.o_stats.mean_tc.mean().cpu().data.numpy()))
        logs.append((prefix + "stats_o/std", self.o_stats.std_tc.mean().cpu().data.numpy()))
        logs.append((prefix + "stats_g/mean", self.g_stats.mean_tc.mean().cpu().data.numpy()))
        logs.append((prefix + "stats_g/std", self.g_stats.std_tc.mean().cpu().data.numpy()))
        return logs

    def __getstate__(self):
        """
        Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        state = {k: v for k, v in self.init_args.items() if not k == "self"}
        # state["shaping"] = self.shaping
        state["tc"] = {
            "o_stats": self.o_stats.state_dict(),
            "g_stats": self.g_stats.state_dict(),
            "policy": self.policy.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }
        return state

    def __setstate__(self, state):
        state_dicts = state.pop("tc")
        self.__init__(**state)
        self.o_stats.load_state_dict(state_dicts["o_stats"])
        self.g_stats.load_state_dict(state_dicts["g_stats"])
        self.policy.load_state_dict(state_dicts["policy"])
        self.policy_optimizer.load_state_dict(state_dicts["policy_optimizer"])