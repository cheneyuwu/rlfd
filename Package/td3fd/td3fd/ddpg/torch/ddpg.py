import pickle

import numpy as np
import torch

from td3fd.memory import RingReplayBuffer, UniformReplayBuffer
from td3fd.ddpg.torch.model import Actor, Critic

# from td3fd.actor_critic import ActorCritic
# from td3fd.demo_shaping import EnsGANDemoShaping, EnsNFDemoShaping
from td3fd.ddpg.torch.normalizer import Normalizer


class DDPG(object):
    def __init__(
        self,
        input_dims,
        use_td3,
        layer_sizes,
        polyak,
        buffer_size,
        batch_size,
        q_lr,
        pi_lr,
        norm_eps,
        norm_clip,
        max_u,
        action_l2,
        clip_obs,
        scope,
        eps_length,
        fix_T,
        clip_pos_returns,
        clip_return,
        sample_demo_buffer,
        batch_size_demo,
        use_demo_reward,
        num_demo,
        demo_strategy,
        bc_params,
        shaping_params,
        gamma,
        info,
        num_epochs,
        num_cycles,
        num_batches,
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
            input_dims         (dict of ints) - dimensions for the observation (o), the goal (g), and the actions (u)
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
        self.num_epochs = num_epochs
        self.num_cycles = num_cycles
        self.num_batches = num_batches
        self.input_dims = input_dims
        self.use_td3 = use_td3
        self.layer_sizes = layer_sizes
        # self.initializer_type = initializer_type
        self.polyak = polyak
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.q_lr = q_lr
        self.pi_lr = pi_lr
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.max_u = max_u
        self.action_l2 = action_l2
        # self.clip_obs = clip_obs
        self.eps_length = eps_length
        self.fix_T = fix_T
        # self.clip_pos_returns = clip_pos_returns
        # self.clip_return = clip_return
        self.sample_demo_buffer = sample_demo_buffer
        self.batch_size_demo = batch_size_demo
        self.use_demo_reward = use_demo_reward
        self.num_demo = num_demo
        self.demo_strategy = demo_strategy
        assert self.demo_strategy in ["none", "bc", "gan", "nf"]
        self.bc_params = bc_params
        self.shaping_params = shaping_params
        self.gamma = gamma
        self.info = info

        # Prepare parameters
        self.dimo = self.input_dims["o"]
        self.dimg = self.input_dims["g"]
        self.dimu = self.input_dims["u"]

        self._create_memory()
        self._create_network()

    def get_actions(self, o, g, compute_q=False):
        o = torch.from_numpy(o)
        g = torch.from_numpy(g)
        o, g = self._normalize_state(o, g)
        u = self.main_actor(o=o, g=g)
        if compute_q:
            q = self.main_critic(o=o, g=g, u=u)
            if self.demo_shaping:
                p = torch.Tensor((0.0,))  # TODO
            else:
                p = q
        u = u * self.max_u
        if compute_q:
            return [u.data.numpy(), p.data.numpy(), q.data.numpy()]
        else:
            return u.data.numpy()

    def init_demo_buffer(self, demo_file, update_stats=True):
        """Initialize the demonstration buffer.
        """
        # load the demonstration data from data file
        episode_batch = self.demo_buffer.load_from_file(data_file=demo_file, num_demo=self.num_demo)
        self._update_demo_stats(episode_batch)
        if update_stats:
            self._update_stats(episode_batch)

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key ('o' and 'ag' is of size T+1, others are of size T)
        """
        self.replay_buffer.store_episode(episode_batch)
        if update_stats:
            self._update_stats(episode_batch)

    def sample_batch(self):
        # use demonstration buffer to sample as well if demo flag is set TRUE
        if self.sample_demo_buffer:
            transitions = {}
            transition_rollout = self.replay_buffer.sample(self.batch_size)
            transition_demo = self.demo_buffer.sample(self.batch_size_demo)
            assert transition_rollout.keys() == transition_demo.keys()
            for k in transition_rollout.keys():
                transitions[k] = np.concatenate((transition_rollout[k], transition_demo[k]))
        else:
            transitions = self.replay_buffer.sample(self.batch_size)  # otherwise only sample from primary buffer
        return transitions

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def save_replay_buffer(self, path):
        pass

    def load_replay_buffer(self, path):
        pass

    def save_weights(self, path):
        pass

    def load_weights(self, path):
        pass

    def train_shaping(self):
        pass

    def train(self):

        batch = self.sample_batch()

        r_tc = torch.from_numpy(batch["r"])
        o_tc, g_tc = self._normalize_state(torch.from_numpy(batch["o"]), torch.from_numpy(batch["g"]))
        o_2_tc, g_2_tc = self._normalize_state(torch.from_numpy(batch["o_2"]), torch.from_numpy(batch["g_2"]))

        u_tc = torch.from_numpy(batch["u"]) / self.max_u
        u_2_tc = self.target_actor(o=o_2_tc, g=g_2_tc)

        # Critic update
        target_tc = r_tc
        if self.use_td3:
            target_tc += self.gamma * torch.min(
                self.target_critic(o=o_2_tc, g=g_2_tc, u=u_2_tc), self.target_critic_twin(o=o_2_tc, g=g_2_tc, u=u_2_tc)
            )
            critic_loss = self.q_criterion(target_tc, self.main_critic(o=o_tc, g=g_tc, u=u_tc))
            critic_twin_loss = self.q_criterion(target_tc, self.main_critic_twin(o=o_tc, g=g_tc, u=u_tc))
        else:
            target_tc += self.gamma * self.target_critic(o=o_2_tc, g=g_2_tc, u=u_2_tc)
            critic_loss = self.q_criterion(target_tc, self.main_critic(o=o_tc, g=g_tc, u=u_tc))

        self.critic_adam.zero_grad()
        critic_loss.backward()
        self.critic_adam.step()
        if self.use_td3:
            self.critic_twin_adam.zero_grad()
            critic_twin_loss.backward()
            self.critic_twin_adam.step()

        # Actor update
        pi_tc = self.main_actor(o=o_tc, g=g_tc)
        actor_loss = -torch.mean(self.main_critic(o=o_tc, g=g_tc, u=pi_tc))
        actor_loss += self.action_l2 * torch.mean(pi_tc)

        self.actor_adam.zero_grad()
        actor_loss.backward()
        self.actor_adam.step()
        return actor_loss.data.numpy(), critic_loss.data.numpy()

    def initialize_target_net(self):
        func = lambda v: v[0].data.copy_(v[1].data)
        map(func, zip(self.target_actor.parameters(), self.main_actor.parameters()))
        map(func, zip(self.target_critic.parameters(), self.main_critic.parameters()))
        if self.use_td3:
            map(func, zip(self.target_critic_twin.parameters(), self.main_critic_twin.parameters()))

    def update_target_net(self):
        func = lambda v: v[0].data.copy_(self.polyak * v[0].data + (1.0 - self.polyak) * v[1].data)
        map(func, zip(self.target_actor.parameters(), self.main_actor.parameters()))
        map(func, zip(self.target_critic.parameters(), self.main_critic.parameters()))
        if self.use_td3:
            map(func, zip(self.target_critic_twin.parameters(), self.main_critic_twin.parameters()))

    def logs(self, prefix=""):
        logs = []
        logs.append((prefix + "stats_o/mean", self.o_stats.mean_tc.numpy()))
        logs.append((prefix + "stats_o/std", self.o_stats.std_tc.numpy()))
        if self.dimg != 0:
            logs.append((prefix + "stats_g/mean", self.g_stats.mean_tc.numpy()))
            logs.append((prefix + "stats_g/std", self.g_stats.std_tc.numpy()))
        return logs

    def _create_memory(self):
        # buffer shape
        buffer_shapes = {}
        if self.fix_T:
            buffer_shapes["o"] = (self.eps_length + 1, self.dimo)
            buffer_shapes["u"] = (self.eps_length, self.dimu)
            buffer_shapes["r"] = (self.eps_length, 1)
            if self.dimg != 0:  # for multigoal environment - or states that do not change over episodes.
                buffer_shapes["ag"] = (self.eps_length + 1, self.dimg)
                buffer_shapes["g"] = (self.eps_length, self.dimg)
            for key, val in self.input_dims.items():
                if key.startswith("info"):
                    buffer_shapes[key] = (self.eps_length, *(tuple([val]) if val > 0 else tuple()))
        else:
            buffer_shapes["o"] = (self.dimo,)
            buffer_shapes["o_2"] = (self.dimo,)
            buffer_shapes["u"] = (self.dimu,)
            buffer_shapes["r"] = (1,)
            if self.dimg != 0:  # for multigoal environment - or states that do not change over episodes.
                buffer_shapes["ag"] = (self.dimg,)
                buffer_shapes["g"] = (self.dimg,)
                buffer_shapes["ag_2"] = (self.dimg,)
                buffer_shapes["g_2"] = (self.dimg,)
            for key, val in self.input_dims.items():
                if key.startswith("info"):
                    buffer_shapes[key] = tuple([val]) if val > 0 else tuple()
            # need the "done" signal for restarting from training
            buffer_shapes["done"] = (1,)
        # initialize replay buffer(s)
        if self.fix_T:
            self.replay_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
            if self.demo_strategy != "none" or self.sample_demo_buffer:
                self.demo_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
        else:
            self.replay_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)
            if self.demo_strategy != "none" or self.sample_demo_buffer:
                self.demo_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)

    def _create_network(self):
        # Normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip)

        # Models
        self.main_actor = Actor(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, layer_sizes=self.layer_sizes, noise=False
        )
        self.target_actor = Actor(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, layer_sizes=self.layer_sizes, noise=self.use_td3
        )
        self.actor_adam = torch.optim.Adam(self.main_actor.parameters(), lr=self.pi_lr)

        self.main_critic = Critic(dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, layer_sizes=self.layer_sizes)
        self.target_critic = Critic(dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, layer_sizes=self.layer_sizes)
        self.critic_adam = torch.optim.Adam(self.main_critic.parameters(), lr=self.q_lr)
        if self.use_td3:
            self.main_critic_twin = Critic(
                dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, layer_sizes=self.layer_sizes
            )
            self.target_critic_twin = Critic(
                dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, layer_sizes=self.layer_sizes
            )
            self.critic_twin_adam = torch.optim.Adam(self.main_critic_twin.parameters(), lr=self.q_lr)

        self.demo_shaping = None  # TODO

        self.q_criterion = torch.nn.MSELoss()

        self.initialize_target_net()

    def _normalize_state(self, o, g):
        o = self.o_stats.normalize(o)
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            g = self.g_stats.normalize(g)
        return o, g

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

        self.o_stats.update(torch.from_numpy(transitions["o"]))
        if self.dimg != 0:
            self.g_stats.update(torch.from_numpy(transitions["g"]))

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

        self.demo_o_stats.update(torch.from_numpy(transitions["o"]))
        if self.dimg != 0:
            self.demo_g_stats.update(torch.from_numpy(transitions["g"]))

    def __getstate__(self):
        pass
        # """
        # Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        # """
        # state = {k: v for k, v in self.init_args.items() if not k == "self"}
        # state["tf"] = self.sess.run([x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        # return state

    def __setstate__(self, state):
        pass
        # kwargs = state["kwargs"]
        # del state["kwargs"]

        # self.__init__(**state, **kwargs)
        # vars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        # assert len(vars) == len(state["tf"])
        # node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        # self.sess.run(node)
