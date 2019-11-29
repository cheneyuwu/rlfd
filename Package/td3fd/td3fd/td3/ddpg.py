import pickle

import numpy as np
import torch
import torch.nn.functional as F

from td3fd.td3.actorcritic_network import Actor, Critic
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
        norm_eps,
        norm_clip,
        # TODO
        demo_strategy,
        bc_params,
        info,
        fix_T,
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

        self.norm_eps = norm_eps
        self.norm_clip = norm_clip

        self.action_l2 = action_l2

        self.demo_strategy = demo_strategy
        assert self.demo_strategy in ["bc", "gan", "nf", "none"]
        self.bc_params = bc_params
        self.gamma = gamma
        self.gamma = 0.99  # TODO make this a parameter
        self.info = info

        # self.fix_T = fix_T

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
        summary(self.main_actor, [self.dimo, self.dimg])
        # critic
        self.main_critic = Critic(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
        self.target_critic = Critic(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
        self.critic_optimizer = torch.optim.Adam(self.main_critic.parameters(), lr=self.q_lr)
        summary(self.main_critic, [self.dimo, self.dimg, self.dimu])
        # critic twin
        if self.twin_delayed:
            self.main_critic_twin = Critic(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
            self.target_critic_twin = Critic(self.dimo, self.dimg, self.dimu, self.max_u, self.layer_sizes).to(device)
            self.critic_twin_optimizer = torch.optim.Adam(self.main_critic_twin.parameters(), lr=self.q_lr)
            summary(self.main_critic_twin, [self.dimo, self.dimg, self.dimu])

        self.shaping = None  # TODO
        self.initialize_target_net()
        self.total_it = 0

    def get_actions(self, o, g, compute_q=False):
        o = o.reshape((-1, *self.dimo))
        g = g.reshape((o.shape[0], *self.dimg))
        o_tc = torch.FloatTensor(o).to(device)
        g_tc = torch.FloatTensor(g).to(device)
        u_tc = self.main_actor(o=o_tc, g=g_tc)
        # TODO
        # o_tc = self.o_stats.normalize(torch.FloatTensor(o).to(device))
        # g_tc = self.g_stats.normalize(torch.FloatTensor(g).to(device))

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

        r_tc = torch.FloatTensor(batch["r"]).to(device)
        o_tc = torch.FloatTensor(batch["o"]).to(device)
        g_tc = torch.FloatTensor(batch["g"]).to(device)
        u_tc = torch.FloatTensor(batch["u"]).to(device)
        o_2_tc = torch.FloatTensor(batch["o_2"]).to(device)
        g_2_tc = torch.FloatTensor(batch["g_2"]).to(device)
        done_tc = torch.FloatTensor(batch["done"]).to(device)
        if demo_batch != None:  ## Behavior cloning
            do_tc = torch.FloatTensor(demo_batch["o"]).to(device)
            dg_tc = torch.FloatTensor(demo_batch["g"]).to(device)
            du_tc = torch.FloatTensor(demo_batch["u"]).to(device)
        # TODO
        # o_tc = self.o_stats.normalize(torch.FloatTensor(batch["o"]).to(device))
        # g_tc = self.g_stats.normalize(torch.FloatTensor(batch["g"]).to(device))
        # o_2_tc = self.o_stats.normalize(torch.FloatTensor(batch["o_2"]).to(device))
        # g_2_tc = self.g_stats.normalize(torch.FloatTensor(batch["g_2"]).to(device))

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(u_tc) * self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
            u_2_tc = (self.target_actor(o=o_2_tc, g=g_2_tc) + noise * self.max_u).clamp(-self.max_u, self.max_u)
            target_tc = r_tc
            if self.demo_strategy in ["gan", "nf"]:  # add reward shaping
                assert self.shaping != None
                target_tc += self.shaping.reward(o_tc, g_tc, u_tc, o_2_tc, g_2_tc, u_2_tc)
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
        actor_loss += self.action_l2 * torch.mean(pi_tc)
        if self.demo_strategy in ["gan", "nf"]:
            assert self.shaping != None
            actor_loss += torch.mean(self.shaping.potential(o_tc, g_tc, u_tc))
            dpi_tc = self.main_actor(o=do_tc, g=dg_tc)
            actor_loss += -torch.mean(self.main_critic(o=do_tc, g=dg_tc, u=dpi_tc))
            actor_loss += self.action_l2 * torch.mean(dpi_tc)
            actor_loss += torch.mean(self.shaping.potential(do_tc, dg_tc, du_tc))
        if self.demo_strategy == "bc":
            dpi_tc = self.main_actor(o=do_tc, g=dg_tc)
            actor_loss += -torch.mean(self.main_critic(o=do_tc, g=dg_tc, u=dpi_tc))
            actor_loss += self.action_l2 * torch.mean(dpi_tc)
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
        self.o_stats.update(torch.FloatTensor(batch["o"]))
        self.g_stats.update(torch.FloatTensor(batch["g"]))

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
        self.__init__(**state)
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
