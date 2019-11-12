"""Adopted from OpenAI baselines, HER
"""
import pickle
from collections import deque

import numpy as np

from td3fd import logger

try:
    from mujoco_py import MujocoException
except:
    MujocoException = None


class RolloutWorkerBase:
    def __init__(
        self,
        make_env,
        policy,
        dims,
        eps_length,
        max_u,
        noise_eps,
        polyak_noise,
        random_eps,
        rollout_batch_size,
        compute_q,
        history_len,
        render,
        **kwargs
    ):
        """
        Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env           (function)    - a factory function that creates a new instance of the environment when called
            policy             (object)      - the policy that is used to act
            dims               (dict of int) - the dimensions for observations (o), goals (g), and actions (u)
            rollout_batch_size (int)         - the number of parallel rollouts that should be used
            compute_q          (bool)        - whether or not to compute the Q values alongside the actions
            noise_eps          (float)       - scale of the additive Gaussian noise
            random_eps         (float)       - probability of selecting a completely random action
            history_len        (int)         - length of history for statistics smoothing
            render             (bool)        - whether or not to render the rollouts
        """
        # Parameters
        self.make_env = make_env
        self.policy = policy
        self.dims = dims
        self.eps_length = eps_length
        self.max_u = max_u
        self.rollout_batch_size = rollout_batch_size
        self.compute_q = compute_q
        self.noise_eps = noise_eps
        self.polyak_noise = polyak_noise
        self.random_eps = random_eps
        self.render = render

        assert self.eps_length > 0
        self.info_keys = [key.replace("info_", "") for key in dims.keys() if key.startswith("info_")]

        self.history = {
            "success_history": deque(maxlen=history_len),
            "total_reward_history": deque(maxlen=history_len),
            "total_shaping_reward_history": deque(maxlen=history_len),
            "Q_history": deque(maxlen=history_len),
            "QP_history": deque(maxlen=history_len),
            "n_episodes": 0,
        }

    def load_history_from_file(self, path):
        """
        store the history into a npz file
        """
        with open(path, "rb") as f:
            self.history = pickle.load(f)

    def dump_history_to_file(self, path):
        """
        store the history into a npz file
        """
        with open(path, "wb") as f:
            pickle.dump(self.history, f)

    def seed(self, seed):
        """set seed for environment
        """
        raise NotImplementedError

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts for maximum time horizon `eps_length` with the current policy
        """
        raise NotImplementedError

    def reset(self):
        """Perform a reset of environments
        """
        raise NotImplementedError

    def logs(self, prefix="worker"):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [("total_reward", np.mean(self.history["total_reward_history"]))]
        logs += [("total_shaping_reward", np.mean(self.history["total_shaping_reward_history"]))]
        logs += [("success_rate", np.mean(self.history["success_history"]))]
        if self.compute_q:
            logs += [("mean_Q", np.mean(self.history["Q_history"]))]
            logs += [("mean_Q_plus_P", np.mean(self.history["QP_history"]))]
        logs += [("episode", self.history["n_episodes"])]

        if prefix is not "" and not prefix.endswith("/"):
            return [(prefix + "/" + key, val) for key, val in logs]
        else:
            return logs

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        for k, v in self.history.items():
            if k == "n_episodes":
                continue
            v.clear()

    def current_total_reward(self):
        return np.mean(self.history["total_reward_history"])

    def current_total_shaping_reward(self):
        return np.mean(self.history["total_shaping_reward_history"])

    def current_success_rate(self):
        return np.mean(self.history["success_history"])

    def current_mean_Q(self):
        return np.mean(self.history["Q_history"])

    def current_mean_QP(self):
        return np.mean(self.history["QP_history"])

    def _add_noise_to_action(self, u):
        # update noise
        assert (
            not self.polyak_noise
            or (type(self.last_noise) == float and self.last_noise == 0.0)
            or self.last_noise.shape == u.shape
        )
        last_noise = self.polyak_noise * self.last_noise
        gauss_noise = (1.0 - self.polyak_noise) * self.noise_eps * self.max_u * np.random.randn(*u.shape)
        self.last_noise = gauss_noise + last_noise
        # add noise to u
        u = u + self.last_noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, self.random_eps, u.shape[0]).reshape(-1, 1) * (
            self._random_action(u.shape[0]) - u
        )  # eps-greedy
        # assert u.shape[0] != 1, "the output is 1, make sure that the code behaves correctly (u = u[0]?)"
        return u

    def _clear_noise_history(self):
        self.last_noise = 0.0

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, *self.dims["u"]))


class RolloutWorker(RolloutWorkerBase):
    def __init__(
        self,
        make_env,
        policy,
        dims,
        eps_length,
        max_u,
        noise_eps=0.0,
        polyak_noise=0.0,
        random_eps=0.0,
        rollout_batch_size=1,
        compute_q=False,
        history_len=10,
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
            compute_q          (bool)        - whether or not to compute the Q values alongside the actions
            noise_eps          (float)       - scale of the additive Gaussian noise
            random_eps         (float)       - probability of selecting a completely random action
            history_len        (int)         - length of history for statistics smoothing
            render             (bool)        - whether or not to render the rollouts
        """
        super().__init__(
            make_env=make_env,
            policy=policy,
            dims=dims,
            eps_length=eps_length,
            max_u=max_u,
            noise_eps=noise_eps,
            polyak_noise=polyak_noise,
            random_eps=random_eps,
            rollout_batch_size=rollout_batch_size,
            compute_q=compute_q,
            history_len=history_len,
            render=render,
        )

        self.envs = [self.make_env() for _ in range(self.rollout_batch_size)]
        self.initial_o = np.empty((self.rollout_batch_size, *self.dims["o"]), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, *self.dims["g"]), np.float32)  # achieved goals
        self.g = np.empty((self.rollout_batch_size, *self.dims["g"]), np.float32)  # goals

        self.reset()
        self.clear_history()

    def seed(self, seed):
        """ Set seed for environment
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)

    def generate_rollouts(self):
        """ Performs `rollout_batch_size` rollouts for maximum time horizon `eps_length` with the current policy
        """

        # Information to store
        obs, achieved_goals, acts, goals, rewards, successes, shaping_rewards, dones = [], [], [], [], [], [], [], []
        policy_rewards, policy_values = [], []  # rewards and values from policy
        info_values = [
            np.empty((self.eps_length, self.rollout_batch_size, *self.dims["info_" + key]), np.float32)
            for key in self.info_keys
        ]
        Qs, QPs = [], []

        # Store initial observations and goals
        self.reset()
        # Clear noise history for polyak noise
        self._clear_noise_history()

        o = np.empty((self.rollout_batch_size, *self.dims["o"]), np.float32)  # o
        ag = np.empty((self.rollout_batch_size, *self.dims["g"]), np.float32)  # ag
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # Main episode loop
        for t in range(self.eps_length):
            # get the action for all envs of the current batch
            policy_output = self.policy.get_actions(o, self.g, compute_q=self.compute_q)
            if self.compute_q:
                u = self._add_noise_to_action(policy_output[0])
                # Q value
                Q = policy_output[1]
                Q = Q.reshape(-1, 1)
                # Q plus P
                QP = policy_output[2]
                QP = QP.reshape(-1, 1)
            else:
                u = self._add_noise_to_action(policy_output)
            # compute policy estimated reward
            pr = self.policy.get_rewards(o, self.g, u).reshape(-1, 1)

            u = u.reshape(-1, *self.dims["u"])  # make sure that the shape is correct
            # compute the next states
            o_new = np.empty((self.rollout_batch_size, *self.dims["o"]))  # o_2
            ag_new = np.empty((self.rollout_batch_size, *self.dims["g"]))  # ag_2
            r = np.empty((self.rollout_batch_size, 1))  # reward
            success = np.zeros(self.rollout_batch_size)  # from info
            shaping_reward = np.zeros((self.rollout_batch_size, 1))  # from info
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    curr_o_new, curr_r, _, info = self.envs[i].step(u[i])
                    r[i] = curr_r
                    if "is_success" in info:
                        success[i] = info["is_success"]
                    if "shaping_reward" in info:
                        shaping_reward[i] = info["shaping_reward"]
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
            # compute policy estimated value
            pv = self.policy.get_values(o, self.g).reshape(-1, 1)

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            goals.append(self.g.copy())
            acts.append(u.copy())
            rewards.append(r.copy())
            # policy estimated rewards and values
            policy_rewards.append(pr.copy())
            policy_values.append(pv.copy())
            successes.append(success.copy())
            shaping_rewards.append(shaping_reward.copy())
            # extra done signal to indicate the episode is done
            dones.append(np.array((self.rollout_batch_size * [float(t == self.eps_length - 1)])).reshape(-1, 1))
            if self.compute_q:
                Qs.append(Q.copy())
                QPs.append(QP.copy())
            o[...] = o_new  # o_2 -> o
            ag[...] = ag_new  # ag_2 -> ag
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        pv = self.policy.get_values(o, self.g).reshape(-1, 1)
        policy_values.append(pv.copy())

        # Store all information into an episode dict
        episode = dict(
            o=obs, u=acts, g=goals, ag=achieved_goals, r=rewards, pr=policy_rewards, pv=policy_values, done=dones
        )
        for key, value in zip(self.info_keys, info_values):
            episode["info_{}".format(key)] = value
        episode = self._convert_episode_to_batch_major(episode)

        # Store stats
        # success rate
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.history["success_history"].append(success_rate)
        # shaping reward
        total_shaping_rewards = np.sum(np.array(shaping_rewards), axis=0).reshape(-1)
        assert total_shaping_rewards.shape == (self.rollout_batch_size,), total_shaping_rewards.shape
        total_shaping_reward = np.mean(total_shaping_rewards)
        self.history["total_shaping_reward_history"].append(total_shaping_reward)
        # total reward
        total_rewards = np.sum(np.array(rewards), axis=0).reshape(-1)
        assert total_rewards.shape == (self.rollout_batch_size,), total_rewards.shape
        total_reward = np.mean(total_rewards)
        self.history["total_reward_history"].append(total_reward)
        # Q output from critic networks
        if self.compute_q:
            self.history["Q_history"].append(np.mean(Qs))
            self.history["QP_history"].append(np.mean(QPs))
        self.history["n_episodes"] += self.rollout_batch_size

        return episode

    def reset(self):
        """ Perform a reset of environments.
        """
        for i in range(self.rollout_batch_size):
            obs = self.envs[i].reset()
            self.initial_o[i] = obs["observation"]
            self.initial_ag[i] = obs["achieved_goal"]
            self.g[i] = obs["desired_goal"]

    def _convert_episode_to_batch_major(self, episode):
        """ Converts an episode to have the batch dimension in the major (first) dimension.
        """
        episode_batch = {}
        for key in episode.keys():
            val = np.array(episode[key]).copy()
            # make inputs batch-major instead of time-major
            episode_batch[key] = val.swapaxes(0, 1)

        return episode_batch


class SerialRolloutWorker(RolloutWorkerBase):
    def __init__(
        self,
        make_env,
        policy,
        dims,
        eps_length,
        max_u,
        noise_eps=0.0,
        polyak_noise=0.0,
        random_eps=0.0,
        rollout_batch_size=1,
        compute_q=False,
        history_len=10,
        render=False,
        **kwargs
    ):
        """
        Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env           (func)        - a factory function that creates a new instance of the environment when called
            policy             (cls)         - the policy that is used to act
            dims               (dict of int) - the dimensions for observations (o), goals (g), and actions (u)
            rollout_batch_size (int)         - the number of parallel rollouts that should be used
            compute_q          (bool)        - whether or not to compute the Q values alongside the actions
            noise_eps          (float)       - scale of the additive Gaussian noise
            random_eps         (float)       - probability of selecting a completely random action
            history_len        (int)         - length of history for statistics smoothing
            render             (bool)        - whether or not to render the rollouts
        """
        super().__init__(
            make_env=make_env,
            policy=policy,
            dims=dims,
            eps_length=eps_length,
            max_u=max_u,
            noise_eps=noise_eps,
            polyak_noise=polyak_noise,
            random_eps=random_eps,
            rollout_batch_size=rollout_batch_size,
            compute_q=compute_q,
            history_len=history_len,
            render=render,
        )

        # create env and initial os and gs
        self.env = make_env()
        self.initial_o = np.empty(self.dims["o"], np.float32)  # observations
        self.initial_ag = np.empty(self.dims["g"], np.float32)  # achieved goals
        self.g = np.empty(self.dims["g"], np.float32)  # goals

        self.reset()
        self.clear_history()

    def seed(self, seed):
        """ Set seed for environment
        """
        self.env.seed(seed)

    def generate_rollouts(self):
        """ Performs `rollout_batch_size` rollouts for maximum time horizon `eps_length` with the current policy
        """

        # Information to store
        obs, obs_2, achieved_goals, achieved_goals_2, acts, goals, rewards = [], [], [], [], [], [], []
        policy_rewards, policy_values, policy_values_2 = [], [], []  # rewards and values from policy
        successes, shaping_rewards = [], []
        info_values = {"info_" + k: [] for k in self.info_keys}
        Qs, QPs = [], []
        dones = []

        for batch in range(self.rollout_batch_size):

            self.reset()

            # Clear noise history for polyak noise
            self._clear_noise_history()

            # Store initial observations and goals
            o = np.empty(self.dims["o"], np.float32)  # o
            ag = np.empty(self.dims["g"], np.float32)  # ag
            o[...] = self.initial_o
            ag[...] = self.initial_ag

            # Main episode loop
            for t in range(self.eps_length):
                # get the action for all envs of the current batch
                policy_output = self.policy.get_actions(o, self.g, compute_q=self.compute_q)
                if self.compute_q:
                    u = self._add_noise_to_action(policy_output[0])
                    # Q value
                    Q = policy_output[1]
                    Q = Q.reshape(1)
                    # Q plus P
                    QP = policy_output[2]
                    QP = QP.reshape(1)
                else:
                    u = self._add_noise_to_action(policy_output)
                # compute policy estimated reward
                pr = self.policy.get_rewards(o, self.g, u).reshape((1,))

                u = u.reshape(self.dims["u"])  # make sure that the shape is correct
                # compute the next states
                o_new = np.empty(self.dims["o"])  # o_2
                ag_new = np.empty(self.dims["g"])  # ag_2
                r = np.empty((1,))  # reward
                success = np.zeros((1,))  # from info
                shaping_reward = np.zeros((1,))  # from info
                iv = {"info_" + k: np.empty(self.dims["info_" + k]) for k in self.info_keys}
                # compute new states and observations
                try:
                    curr_o_new, curr_r, _, info = self.env.step(u)
                    o_new[...] = curr_o_new["observation"]
                    ag_new[...] = curr_o_new["achieved_goal"]
                    r[...] = curr_r
                    if "is_success" in info:
                        success[...] = info["is_success"]
                    if "shaping_reward" in info:
                        shaping_reward[...] = info["shaping_reward"]
                    for key in self.info_keys:
                        iv["info_" + key][...] = info[key]
                    if self.render:
                        self.env.render()
                except MujocoException:
                    logger.warn("MujocoException caught during rollout generation. Trying again...")
                    return self.generate_rollouts()
                if np.isnan(o_new).any():
                    logger.warn("NaN caught during rollout generation. Trying again...")
                    return self.generate_rollouts()

                # compute policy estimated value
                pv = self.policy.get_values(o, self.g).reshape((1,))
                pv_new = self.policy.get_values(o_new, self.g).reshape((1,))

                #
                obs.append(o.copy())
                obs_2.append(o_new.copy())
                achieved_goals.append(ag.copy())
                achieved_goals_2.append(ag_new.copy())
                acts.append(u.copy())
                goals.append(self.g.copy())
                rewards.append(r.copy())
                # policy estimated rewards and values
                policy_rewards.append(pr.copy())
                policy_values.append(pv.copy())
                policy_values_2.append(pv_new.copy())
                for key in self.info_keys:
                    info_values["info_" + key].append(iv["info_" + key].copy())

                # extra plotting info recorded per step
                shaping_rewards.append(shaping_reward.copy())
                if self.compute_q:
                    Qs.append(Q.copy())
                    QPs.append(QP.copy())

                o[...] = o_new  # o_2 -> o
                ag[...] = ag_new  # ag_2 -> ag

                # UNCOMMENT this to end this episode if succeeded
                if success:
                    if t == 0:
                        logger.warn("Starting with a success, this may be an indication of error!")
                    dones.append(np.ones(1))
                    break

                dones.append(np.array((float(t == self.eps_length - 1),)))

            # extra plotting into recorded per episode
            successes.append(success.copy())

        # Store all information into an episode dict
        episode = dict(
            o=obs,
            o_2=obs_2,
            ag=achieved_goals,
            ag_2=achieved_goals_2,
            u=acts,
            g=goals,
            g_2=goals,
            r=rewards,
            pr=policy_rewards,
            pv=policy_values,
            pv_2=policy_values_2,
            done=dones,
        )
        for key, value in info_values.items():
            episode[key] = value
        for key, value in episode.items():
            episode[key] = np.array(episode[key])

        # Store stats
        # success rate
        success_rate = np.sum(np.array(successes)) / self.rollout_batch_size
        self.history["success_history"].append(success_rate)
        # shaping reward
        total_shaping_reward = np.sum(np.array(shaping_rewards)) / self.rollout_batch_size
        self.history["total_shaping_reward_history"].append(total_shaping_reward)
        # total reward
        total_reward = np.sum(np.array(rewards)) / self.rollout_batch_size
        self.history["total_reward_history"].append(total_reward)
        # Q output from critic networks
        if self.compute_q:
            self.history["Q_history"].append(np.mean(Qs))
            self.history["QP_history"].append(np.mean(QPs))
        self.history["n_episodes"] += self.rollout_batch_size

        return episode

    def reset(self):
        """ Perform a reset of environments
        """
        obs = self.env.reset()
        self.initial_o[...] = obs["observation"]
        self.initial_ag[...] = obs["achieved_goal"]
        self.g[...] = obs["desired_goal"]
