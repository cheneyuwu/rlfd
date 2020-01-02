import pickle

import numpy as np
import torch
import torch.nn.functional as F

# TODO switch between networks for image input and state input. image assumed to be (3, 32, 32)
from td3fd.td3.actorcritic_network import Actor, Critic

# from td3fd.td3.actorcritic_network_img import Actor, Critic # this is for image

from td3fd.td3.normalizer import Normalizer
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG(object):
    def __init__(
        self,
        # for learning
        num_epochs,
        num_cycles,
        num_batches,
        batch_size,
        batch_size_demo,
        # configuration
        dims,
        layer_sizes,
        twin_delayed,
        policy_freq,
        policy_noise,
        policy_noise_clip,
        q_lr,
        pi_lr,
        max_u,
        polyak,
        gamma,
        action_l2,
        norm_obs,
        norm_eps,
        norm_clip,
        # TODO
        demo_strategy,
        bc_params,
        info,
    ):
        # Store initial args passed into the function
        self.init_args = locals()

        # Parameters
        self.dims = dims
        self.layer_sizes = layer_sizes
        self.twin_delayed = twin_delayed
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.polyak = polyak
        self.q_lr = q_lr
        self.pi_lr = pi_lr
        self.max_u = max_u

        self.norm_obs = norm_obs
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip

        self.action_l2 = action_l2

        self.demo_strategy = demo_strategy
        assert self.demo_strategy in ["bc", "gan", "nf", "none"]
        self.bc_params = bc_params
        self.gamma = gamma
        self.info = info

        # Prepare parameters
        self.dimo = self.dims["o"]
        self.dimg = self.dims["g"]
        self.dimu = self.dims["u"]

        # Normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip).to(device)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip).to(device)

        # Models
        # actor
        self.main_actor = Actor(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
        self.target_actor = Actor(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
        self.actor_optimizer = torch.optim.Adam(self.main_actor.parameters(), lr=self.pi_lr)
        # summary(self.main_actor, [self.dimo, self.dimg])
        # critic
        self.main_critic = Critic(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
        self.target_critic = Critic(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
        self.critic_optimizer = torch.optim.Adam(self.main_critic.parameters(), lr=self.q_lr)
        # summary(self.main_critic, [self.dimo, self.dimg, self.dimu])
        # critic twin
        if self.twin_delayed:
            self.main_critic_twin = Critic(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
            self.target_critic_twin = Critic(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
            self.critic_twin_optimizer = torch.optim.Adam(self.main_critic_twin.parameters(), lr=self.q_lr)
            # summary(self.main_critic_twin, [self.dimo, self.dimg, self.dimu])

        self.shaping = None  # TODO
        self.initialize_target_net()
        self.total_it = 0

    def get_actions(self, o, g, compute_q=False):
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

        u_tc = self.main_actor(o=o_tc, g=g_tc)

        if compute_q:
            q = self.main_critic(o=o_tc, g=g_tc, u=u_tc).cpu().data.numpy()
            p = 0.0
            if self.shaping != None:
                p = self.shaping.potential(o_tc, g_tc, u_tc).cpu().data.numpy()

        u = u_tc.cpu().data.numpy()
        if o.shape[0] == 1:
            u = u[0]

        if compute_q:
            return [u, q, p + q]
        else:
            return u

    def save(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def train(self, batch, demo_batch=None):

        self.total_it += 1

        o_tc = torch.tensor(batch["o"], dtype=torch.float).to(device)
        g_tc = torch.tensor(batch["g"], dtype=torch.float).to(device)
        o_2_tc = torch.tensor(batch["o_2"], dtype=torch.float).to(device)
        g_2_tc = torch.tensor(batch["g_2"], dtype=torch.float).to(device)
        u_tc = torch.tensor(batch["u"], dtype=torch.float).to(device)
        r_tc = torch.tensor(batch["r"], dtype=torch.float).to(device)
        done_tc = torch.tensor(batch["done"], dtype=torch.float).to(device)
        if demo_batch != None:
            do_tc = torch.tensor(demo_batch["o"], dtype=torch.float).to(device)
            dg_tc = torch.tensor(demo_batch["g"], dtype=torch.float).to(device)
            du_tc = torch.tensor(demo_batch["u"], dtype=torch.float).to(device)

        # Shaping input which are not normalized
        shaping_o_tc = o_tc.clone().detach()
        shaping_g_tc = g_tc.clone().detach()
        shaping_o_2_tc = o_2_tc.clone().detach()
        shaping_g_2_tc = g_2_tc.clone().detach()
        if demo_batch != None:
            shaping_do_tc = do_tc.clone().detach()
            shaping_dg_tc = dg_tc.clone().detach()

        # Normalize states and actions
        if self.norm_obs:
            o_tc = self.o_stats.normalize(o_tc)
            g_tc = self.g_stats.normalize(g_tc)
            o_2_tc = self.o_stats.normalize(o_2_tc)
            g_2_tc = self.g_stats.normalize(g_2_tc)
            if demo_batch != None:  # Behavior cloning
                do_tc = self.o_stats.normalize(do_tc)
                dg_tc = self.g_stats.normalize(dg_tc)

        # TODO: uncomment for images
        # o_tc = (o_tc * 2.0 / 255.0) - 1.0
        # o_2_tc = (o_2_tc * 2.0 / 255.0) - 1.0

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(u_tc) * self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
            u_2_tc = (self.target_actor(o=o_2_tc, g=g_2_tc) + noise * self.max_u).clamp(-self.max_u, self.max_u)
            target_tc = r_tc
            if self.demo_strategy in ["gan", "nf"]:  # add reward shaping
                assert self.shaping != None
                target_tc += self.shaping.reward(
                    shaping_o_tc, shaping_g_tc, u_tc, shaping_o_2_tc, shaping_g_2_tc, u_2_tc
                )
            if self.twin_delayed:
                target_tc += (
                    (1.0 - done_tc)
                    * self.gamma
                    * torch.min(
                        self.target_critic(o=o_2_tc, g=g_2_tc, u=u_2_tc),
                        self.target_critic_twin(o=o_2_tc, g=g_2_tc, u=u_2_tc),
                    )
                )
            else:
                target_tc += (1.0 - done_tc) * self.gamma * self.target_critic(o=o_2_tc, g=g_2_tc, u=u_2_tc)
        if self.twin_delayed:
            critic_loss = F.mse_loss(target_tc, self.main_critic(o=o_tc, g=g_tc, u=u_tc))
            critic_twin_loss = F.mse_loss(target_tc, self.main_critic_twin(o=o_tc, g=g_tc, u=u_tc))
        else:
            critic_loss = F.mse_loss(target_tc, self.main_critic(o=o_tc, g=g_tc, u=u_tc))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.twin_delayed:
            self.critic_twin_optimizer.zero_grad()
            critic_twin_loss.backward()
            self.critic_twin_optimizer.step()

        if self.total_it % self.policy_freq:
            return

        # Actor update
        pi_tc = self.main_actor(o=o_tc, g=g_tc)
        actor_loss = -torch.mean(self.main_critic(o=o_tc, g=g_tc, u=pi_tc))
        actor_loss += self.action_l2 * torch.mean((pi_tc / self.max_u) ** 2)
        if self.demo_strategy in ["gan", "nf"]:
            assert self.shaping != None
            actor_loss += -torch.mean(self.shaping.potential(shaping_o_tc, shaping_g_tc, u_tc))
            dpi_tc = self.main_actor(o=do_tc, g=dg_tc)
            actor_loss += -torch.mean(self.main_critic(o=do_tc, g=dg_tc, u=dpi_tc))
            actor_loss += self.action_l2 * torch.mean((dpi_tc / self.max_u) ** 2)
            actor_loss += -torch.mean(self.shaping.potential(shaping_do_tc, shaping_dg_tc, du_tc))
        if self.demo_strategy == "bc":
            dpi_tc = self.main_actor(o=do_tc, g=dg_tc)
            actor_loss += -torch.mean(self.main_critic(o=do_tc, g=dg_tc, u=dpi_tc))
            actor_loss += self.action_l2 * torch.mean((dpi_tc / self.max_u) ** 2)
            actor_loss = (
                actor_loss * self.bc_params["prm_loss_weight"]
                + torch.mean((self.main_actor(do_tc, dg_tc) - du_tc) ** 2) * self.bc_params["aux_loss_weight"]
            )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def initialize_target_net(self):
        func = lambda v: v[0].data.copy_(v[1].data)
        list(map(func, zip(self.target_actor.parameters(), self.main_actor.parameters())))
        list(map(func, zip(self.target_critic.parameters(), self.main_critic.parameters())))
        if self.twin_delayed:
            list(map(func, zip(self.target_critic_twin.parameters(), self.main_critic_twin.parameters())))

    def update_target_net(self):
        func = lambda v: v[0].data.copy_(self.polyak * v[0].data + (1.0 - self.polyak) * v[1].data)
        list(map(func, zip(self.target_actor.parameters(), self.main_actor.parameters())))
        list(map(func, zip(self.target_critic.parameters(), self.main_critic.parameters())))
        if self.twin_delayed:
            list(map(func, zip(self.target_critic_twin.parameters(), self.main_critic_twin.parameters())))

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
        state["shaping"] = self.shaping
        state["tc"] = {
            "o_stats": self.o_stats.state_dict(),
            "g_stats": self.g_stats.state_dict(),
            "main_actor": self.main_actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "main_critic": self.main_critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "main_critic_twin": self.main_critic_twin.state_dict(),
            "target_critic_twin": self.target_critic_twin.state_dict(),
            "critic_twin_optimizer": self.critic_twin_optimizer.state_dict(),
        }
        return state

    def __setstate__(self, state):
        state_dicts = state.pop("tc")
        shaping = state.pop("shaping")
        self.__init__(**state)
        self.shaping = shaping
        self.o_stats.load_state_dict(state_dicts["o_stats"])
        self.g_stats.load_state_dict(state_dicts["g_stats"])
        self.main_actor.load_state_dict(state_dicts["main_actor"])
        self.target_actor.load_state_dict(state_dicts["target_actor"])
        self.actor_optimizer.load_state_dict(state_dicts["actor_optimizer"])
        self.main_critic.load_state_dict(state_dicts["main_critic"])
        self.target_critic.load_state_dict(state_dicts["target_critic"])
        self.critic_optimizer.load_state_dict(state_dicts["critic_optimizer"])
        self.main_critic_twin.load_state_dict(state_dicts["main_critic_twin"])
        self.target_critic_twin.load_state_dict(state_dicts["target_critic_twin"])
        self.critic_twin_optimizer.load_state_dict(state_dicts["critic_twin_optimizer"])
