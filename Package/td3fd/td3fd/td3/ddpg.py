import pickle

import numpy as np
import torch
import torch.nn.functional as F

# from td3fd.td3.demo_shaping import EnsGANDemoShaping, EnsNFDemoShaping
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
        # demonstrations
        num_demo,
        sample_demo_buffer,
        batch_size_demo,
        use_demo_reward,
        norm_eps,
        norm_clip,
        # TODO
        demo_strategy,
        bc_params,
        info,
        fix_T,
    ):
        """
        Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER). Added functionality
        to use demonstrations for training to Overcome exploration problem.

        Args:
            # Environment I/O and Config
            max_u              (float)        - maximum action magnitude, i.e. actions are in [-max_u, max_u]
            T                  (int)          - the time horizon for rollouts
            fix_T              (bool)         - every episode has fixed length
            clip_obs           (float)        - clip observations before normalization to be in [-clip_obs, clip_obs]
            clip_pos_returns   (boolean)      - whether or not positive returns should be clipped (i.e. clip to 0)
            clip_return        (float)        - clip returns to be in [-clip_return, clip_return]
            # Normalizer
            norm_eps           (float)        - a small value used in the normalizer to avoid numerical instabilities
            norm_clip          (float)        - normalized inputs are clipped to be in [-norm_clip, norm_clip]
            # NN Configuration
            scope              (str)          - the scope used for the TensorFlow graph
            dims         (dict of ints) - dimensions for the observation (o), the goal (g), and the actions (u)
            layer_sizes        (list of ints) - number of units in each hidden layers
            initializer_type   (str)          - initializer of the weight for both policy and critic
            reuse              (boolean)      - whether or not the networks should be reused
            # Replay Buffer
            buffer_size        (int)          - number of transitions that are stored in the replay buffer
            # Dual Network Set
            polyak             (float)        - coefficient for Polyak-averaging of the target network
            # Training
            batch_size         (int)          - batch size for training
            Q_lr               (float)        - learning rate for the Q (critic) network
            pi_lr              (float)        - learning rate for the pi (actor) network
            action_l2          (float)        - coefficient for L2 penalty on the actions
            gamma              (float)        - gamma used for Q learning updates
            # Use demonstration to shape critic or actor
            sample_demo_buffer (int)          - whether or not to sample from demonstration buffer
            batch_size_demo    (int)          - number of samples to be used from the demonstrations buffer, per mpi thread
            use_demo_reward    (int)          - whether or not to assue that demonstration dataset has rewards
            num_demo           (int)          - number of episodes in to be used in the demonstration buffer
            demo_strategy      (str)          - whether or not to use demonstration with different strategies
            bc_params          (dict)
            shaping_params     (dict)
        """
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
        self.sample_demo_buffer = sample_demo_buffer
        self.use_demo_reward = use_demo_reward

        self.num_demo = num_demo
        self.demo_strategy = demo_strategy
        assert self.demo_strategy in ["none", "bc", "gan", "nf", None]
        self.bc_params = bc_params
        self.gamma = gamma
        self.gamma = 0.99  # TODO make this a parameter
        self.info = info

        self.num_epochs = num_epochs
        self.num_cycles = num_cycles
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.batch_size_demo = batch_size_demo

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

        self.demo_shaping = None  # TODO
        self.initialize_target_net()
        self.total_it = 0

    def get_actions(self, o, g, compute_q=False):
        o = o.reshape((-1, *self.dimo))
        g = g.reshape((o.shape[0], *self.dimg))
        o_tc = self.o_stats.normalize(torch.FloatTensor(o).to(device))
        g_tc = self.g_stats.normalize(torch.FloatTensor(g).to(device))
        u_tc = self.main_actor(o=o_tc, g=g_tc)

        if compute_q:
            q_tc = self.main_critic(o=o_tc, g=g_tc, u=u_tc)
            q = q_tc.cpu().data.numpy()
        u = u_tc.cpu().data.numpy()
        if o.shape[0] == 1:
            u = u[0]

        if compute_q:
            return [u, q, q]
        else:
            return u

    # def init_demo_buffer(self, demo_file, update_stats=True):
    #     """Initialize the demonstration buffer.
    #     """
    #     # load the demonstration data from data file
    #     episode_batch = self.demo_buffer.load_from_file(data_file=demo_file, num_demo=self.num_demo)
    #     self._update_demo_stats(episode_batch)
    #     if update_stats:
    #         self._update_stats(episode_batch)

    # def store_episode(self, episode_batch, update_stats=True):
    #     """
    #     episode_batch: array of batch_size x (T or T+1) x dim_key ('o' and 'ag' is of size T+1, others are of size T)
    #     """
    #     self.replay_buffer.store_episode(episode_batch)
    #     if update_stats:
    #         self._update_stats(episode_batch)

    # def sample_batch(self):
    #     # use demonstration buffer to sample as well if demo flag is set TRUE
    #     if self.sample_demo_buffer:
    #         transitions = {}
    #         transition_rollout = self.replay_buffer.sample(self.batch_size)
    #         transition_demo = self.demo_buffer.sample(self.batch_size_demo)
    #         assert transition_rollout.keys() == transition_demo.keys()
    #         for k in transition_rollout.keys():
    #             transitions[k] = np.concatenate((transition_rollout[k], transition_demo[k]))
    #     else:
    #         transitions = self.replay_buffer.sample(self.batch_size)  # otherwise only sample from primary buffer
    #     return transitions

    def save(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def train_shaping(self):
        pass

    def train(self, batch, shaping=None):

        self.total_it += 1

        r_tc = torch.FloatTensor(batch["r"]).to(device)
        o_tc = self.o_stats.normalize(torch.FloatTensor(batch["o"]).to(device))
        g_tc = self.g_stats.normalize(torch.FloatTensor(batch["g"]).to(device))
        u_tc = torch.FloatTensor(batch["u"]).to(device)
        o_2_tc = self.o_stats.normalize(torch.FloatTensor(batch["o_2"]).to(device))
        g_2_tc = self.g_stats.normalize(torch.FloatTensor(batch["g_2"]).to(device))
        done_tc = torch.FloatTensor(batch["done"]).to(device)

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(u_tc) * self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
            u_2_tc = (self.target_actor(o=o_2_tc, g=g_2_tc) + noise * self.max_u).clamp(-self.max_u, self.max_u)
            target_tc = r_tc
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

        # Actor update
        if self.total_it % self.policy_freq == 0:  # TODO this 2 should be a parameter
            pi_tc = self.main_actor(o=o_tc, g=g_tc)
            actor_loss = -torch.mean(self.main_critic(o=o_tc, g=g_tc, u=pi_tc))
            actor_loss += self.action_l2 * torch.mean(pi_tc)

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

    def logs(self, prefix=""):
        logs = []
        logs.append((prefix + "stats_o/mean", self.o_stats.mean_tc.mean().cpu().data.numpy()))
        logs.append((prefix + "stats_o/std", self.o_stats.std_tc.mean().cpu().data.numpy()))
        logs.append((prefix + "stats_g/mean", self.g_stats.mean_tc.mean().cpu().data.numpy()))
        logs.append((prefix + "stats_g/std", self.g_stats.std_tc.mean().cpu().data.numpy()))
        return logs

    # def _create_memory(self):
    #     # buffer shape
    #     buffer_shapes = {}
    #     if self.fix_T:
    #         buffer_shapes["o"] = (self.eps_length + 1, self.dimo)
    #         buffer_shapes["u"] = (self.eps_length, self.dimu)
    #         buffer_shapes["r"] = (self.eps_length, 1)
    #         if self.dimg != 0:  # for multigoal environment - or states that do not change over episodes.
    #             buffer_shapes["ag"] = (self.eps_length + 1, self.dimg)
    #             buffer_shapes["g"] = (self.eps_length, self.dimg)
    #         for key, val in self.dims.items():
    #             if key.startswith("info"):
    #                 buffer_shapes[key] = (self.eps_length, *(tuple([val]) if val > 0 else tuple()))
    #     else:
    #         buffer_shapes["o"] = (self.dimo,)
    #         buffer_shapes["o_2"] = (self.dimo,)
    #         buffer_shapes["u"] = (self.dimu,)
    #         buffer_shapes["r"] = (1,)
    #         if self.dimg != 0:  # for multigoal environment - or states that do not change over episodes.
    #             buffer_shapes["ag"] = (self.dimg,)
    #             buffer_shapes["g"] = (self.dimg,)
    #             buffer_shapes["ag_2"] = (self.dimg,)
    #             buffer_shapes["g_2"] = (self.dimg,)
    #         for key, val in self.dims.items():
    #             if key.startswith("info"):
    #                 buffer_shapes[key] = tuple([val]) if val > 0 else tuple()
    #         # need the "done" signal for restarting from training
    #         buffer_shapes["done"] = (1,)
    #     # initialize replay buffer(s)
    #     if self.fix_T:
    #         self.replay_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
    #         if self.demo_strategy != "none" or self.sample_demo_buffer:
    #             self.demo_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
    #     else:
    #         self.replay_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)
    #         if self.demo_strategy != "none" or self.sample_demo_buffer:
    #             self.demo_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)

    def _update_stats(self, episode_batch):
        # add transitions to normalizer
        if self.fix_T:
            episode_batch["o_2"] = episode_batch["o"][:, 1:, :]
            if self.dimg != 0:
                episode_batch["ag_2"] = episode_batch["ag"][:, :, :]
                episode_batch["g_2"] = episode_batch["g"][:, :, :]
            num_normalizing_transitions = episode_batch["u"].shape[0] * episode_batch["u"].shape[1]
            transitions = self.replay_buffer.sample_transitions(episode_batch, num_normalizing_transitions)
        else:
            transitions = episode_batch.copy()

        self.o_stats.update(torch.FloatTensor(transitions["o"]))
        if self.dimg != 0:
            self.g_stats.update(torch.FloatTensor(transitions["g"]))

    def _update_demo_stats(self, episode_batch):
        # add transitions to normalizer
        if self.fix_T:
            episode_batch["o_2"] = episode_batch["o"][:, 1:, :]
            if self.dimg != 0:
                episode_batch["ag_2"] = episode_batch["ag"][:, :, :]
                episode_batch["g_2"] = episode_batch["g"][:, :, :]
            num_normalizing_transitions = episode_batch["u"].shape[0] * episode_batch["u"].shape[1]
            transitions = self.demo_buffer.sample_transitions(episode_batch, num_normalizing_transitions)
        else:
            transitions = episode_batch.copy()

        self.demo_o_stats.update(torch.FloatTensor(transitions["o"]))
        if self.dimg != 0:
            self.demo_g_stats.update(torch.FloatTensor(transitions["g"]))

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

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

