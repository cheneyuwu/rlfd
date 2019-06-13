from collections import deque

import numpy as np
import pickle

try:
    from mujoco_py import MujocoException
except:
    MujocoException = None


from yw.tool import logger
from yw.util.util import store_args


class RolloutWorker:
    def __init__(
        self,
        make_env,
        policy,
        dims,
        T,
        rollout_batch_size=1,
        exploit=False,
        use_target_net=False,
        compute_Q=False,
        noise_eps=0,
        random_eps=0,
        history_len=100,
        render=False,
        **kwargs
    ):
        """
        Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env           (function)    - a factory function that creates a new instance of the environment when called
            policy             (object)      - the policy that is used to act
            dims               (dict of int) - the dimensions for observations (o), goals (g), and actions (u)
            rollout_batch_size (int)         - the number of parallel rollouts that should be used
            exploit            (bool)        - whether or not to exploit, i.e. to act optimally according to the current policy without any exploration
            use_target_net     (bool)        - whether or not to use the target net for rollouts
            compute_Q          (bool)        - whether or not to compute the Q values alongside the actions
            noise_eps          (float)       - scale of the additive Gaussian noise
            random_eps         (float)       - probability of selecting a completely random action
            history_len        (int)         - length of history for statistics smoothing
            render             (bool)        - whether or not to render the rollouts
        """
        # Parameters
        self.policy = policy
        self.dims = dims
        self.T = T
        self.rollout_batch_size = rollout_batch_size
        self.exploit = exploit
        self.use_target_net = self.use_target_net
        self.compute_Q = compute_Q
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.render = render

        assert self.T > 0

        self.info_keys = [key.replace("info_", "") for key in dims.keys() if key.startswith("info_")]

        self.success_history = deque(maxlen=history_len)
        self.total_reward_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.n_episodes = 0

        self.envs = [make_env() for _ in range(rollout_batch_size)]
        self.initial_o = np.empty((self.rollout_batch_size, self.dims["o"]), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims["g"]), np.float32)  # achieved goals
        self.g = np.empty((self.rollout_batch_size, self.dims["g"]), np.float32)  # goals

        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs["observation"]
        self.initial_ag[i] = obs["achieved_goal"]
        self.g[i] = obs["desired_goal"]

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims["o"]), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims["g"]), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, rewards, successes = [], [], [], [], [], []
        info_values = [
            np.empty((self.T, self.rollout_batch_size, self.dims["info_" + key]), np.float32) for key in self.info_keys
        ]
        Qs = []
        for t in range(self.T):
            logger.debug("RolloutWorker.generate_rollouts -> step {}".format(t))
            policy_output = self.policy.get_actions(
                o,
                ag,
                self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.0,
                random_eps=self.random_eps if not self.exploit else 0.0,
                use_target_net=self.use_target_net,
            )
            if self.compute_Q:
                u = policy_output[0]
                Q = policy_output[1]
                Q = Q.reshape(-1, 1)
            else:
                u = np.array(policy_output)
            u = u.reshape(-1, self.dims["u"])

            o_new = np.empty((self.rollout_batch_size, self.dims["o"]))
            ag_new = np.empty((self.rollout_batch_size, self.dims["g"]))
            r = np.empty((self.rollout_batch_size, 1))  # reward
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    curr_o_new, curr_r, _, info = self.envs[i].step(u[i])
                    r[i] = curr_r
                    if "is_success" in info:
                        success[i] = info["is_success"]
                    o_new[i] = curr_o_new["observation"]
                    ag_new[i] = curr_o_new["achieved_goal"]
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException:
                    logger.warn("MujocoException caught during rollout generation. Trying again...")
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                logger.warn("NaN caught during rollout generation. Trying again...")
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            rewards.append(r.copy())
            if self.compute_Q:
                Qs.append(Q.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        # Will contain episode info
        episode = dict(o=obs, u=acts, g=goals, ag=achieved_goals, r=rewards)
        if self.compute_Q:
            episode["q"] = Qs
        for key, value in zip(self.info_keys, info_values):
            episode["info_{}".format(key)] = value
        episode = RolloutWorker.convert_episode_to_batch_major(episode)

        # Stats
        # success rate
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        # total reward
        total_rewards = np.sum(np.array(rewards), axis=0).reshape(-1)
        assert total_rewards.shape == (self.rollout_batch_size,), total_rewards.shape
        total_reward = np.mean(total_rewards)
        self.total_reward_history.append(total_reward)
        # Q output from critic networks
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return episode

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.total_reward_history.clear()
        self.success_history.clear()
        self.Q_history.clear()

    def current_total_reward(self):
        return np.mean(self.total_reward_history)

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, "wb") as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix="worker"):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [("success_rate", np.mean(self.success_history))]
        logs += [("total_reward", np.mean(self.total_reward_history))]
        if self.compute_Q:
            logs += [("mean_Q", np.mean(self.Q_history))]
        logs += [("episode", self.n_episodes)]

        if prefix is not "" and not prefix.endswith("/"):
            return [(prefix + "/" + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)

    @staticmethod
    def convert_episode_to_batch_major(episode):
        """Converts an episode to have the batch dimension in the major (first)
        dimension.
        """
        episode_batch = {}
        for key in episode.keys():
            val = np.array(episode[key]).copy()
            # make inputs batch-major instead of time-major
            episode_batch[key] = val.swapaxes(0, 1)

        return episode_batch
