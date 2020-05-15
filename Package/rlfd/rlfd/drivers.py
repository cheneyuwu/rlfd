"""Adopted from OpenAI baselines, HER
"""
import abc
import pickle
from collections import deque

import numpy as np
import tensorflow as tf

from rlfd import logger

try:
  from mujoco_py import MujocoException
except:
  MujocoException = None


class Driver(object, metaclass=abc.ABCMeta):

  def __init__(self, make_env, policy, dims, max_u, noise_eps, polyak_noise,
               random_eps, history_len, render, **kwargs):
    """
    Rollout worker generates experience by interacting with one or many environments.

    This base class handles the following:

    Args:
        make_env           (function)    - a factory function that creates a new instance of the environment when called
        policy             (object)      - the policy that is used to act
        dims               (dict of int) - the dimensions for observations (o), goals (g), and actions (u)
        num_episodes       (int)         - the number of parallel rollouts that should be used
        noise_eps          (float)       - scale of the additive Gaussian noise
        random_eps         (float)       - probability of selecting a completely random action
        history_len        (int)         - length of history for statistics smoothing
        render             (bool)        - whether or not to render the rollouts
    """
    # Parameters
    self.make_env = make_env
    self.policy = policy
    self.dims = dims
    self.max_u = max_u
    self.info_keys = [
        key.replace("info_", "")
        for key in dims.keys()
        if key.startswith("info_")
    ]

    self.noise_eps = noise_eps
    self.polyak_noise = polyak_noise
    self.random_eps = random_eps

    self.render = render

    self.history_len = history_len
    self.history = {}
    self.total_num_episodes = 0
    self.total_num_steps = 0

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
    """set seed for internal environments"""
    return self._seed(seed)

  def generate_rollouts(self, observers=(), random=False):
    """Generate experiences
        random - whether or not to use a random policy
    """
    return self._generate_rollouts(observers=observers, random=random)

  def logs(self, prefix="worker"):
    """Generates a dictionary that contains all collected statistics.
    """
    logs = []
    logs += [("episodes", self.total_num_episodes)]
    logs += [("steps", self.total_num_steps)]
    for k, v in self.history.items():
      logs += [(k, np.mean(v))]

    if prefix is not "" and not prefix.endswith("/"):
      return [(prefix + "/" + key, val) for key, val in logs]
    else:
      return logs

  def add_history_key(self, key):
    self.history[key] = deque(maxlen=self.history_len)

  def add_history_keys(self, keys):
    for key in keys:
      self.history[key] = deque(maxlen=self.history_len)

  def clear_history(self):
    """Clears all histories that are used for statistics
    """
    for k, v in self.history.items():
      v.clear()

  def _add_noise_to_action(self, u):
    u = u.reshape(-1, *self.dims["u"])
    # update noise
    assert (not self.polyak_noise or
            (type(self.last_noise) == float and self.last_noise == 0.0) or
            self.last_noise.shape == u.shape)
    last_noise = self.polyak_noise * self.last_noise
    gauss_noise = (1.0 - self.polyak_noise
                  ) * self.noise_eps * self.max_u * np.random.randn(*u.shape)
    self.last_noise = gauss_noise + last_noise
    # add noise to u
    u = u + self.last_noise
    u = np.clip(u, -self.max_u, self.max_u)
    u += np.random.binomial(1, self.random_eps, u.shape[0]).reshape(-1, 1) * (
        self._random_action(u.shape[0]) - u)  # eps-greedy
    if u.shape[0] == 1:
      u = u[0]
    return u

  def _clear_noise_history(self):
    self.last_noise = 0.0

  def _random_action(self, n):
    return np.random.uniform(low=-self.max_u,
                             high=self.max_u,
                             size=(n, *self.dims["u"]))

  @abc.abstractmethod
  def _seed(self, seed):
    """set seed for environment"""

  @abc.abstractmethod
  def _generate_rollouts(self, observers=(), random=False):
    """Performs `num_episodes` rollouts for maximum time horizon `eps_length` with the current policy"""


class StepBasedDriver(Driver):

  def __init__(self,
               make_env,
               policy,
               dims,
               eps_length,
               max_u,
               num_steps=None,
               num_episodes=None,
               noise_eps=0.0,
               polyak_noise=0.0,
               random_eps=0.0,
               history_len=300,
               render=False,
               **kwargs):
    """
    Rollout worker generates experience by interacting with one or many environments.

    Args:
        make_env           (func)        - a factory function that creates a new instance of the environment when called
        policy             (cls)         - the policy that is used to act
        dims               (dict of int) - the dimensions for observations (o), goals (g), and actions (u)
        noise_eps          (float)       - scale of the additive Gaussian noise
        random_eps         (float)       - probability of selecting a completely random action
        history_len        (int)         - length of history for statistics smoothing
        render             (bool)        - whether or not to render the rollouts
    """
    super().__init__(
        make_env=make_env,
        policy=policy,
        dims=dims,
        max_u=max_u,
        noise_eps=noise_eps,
        polyak_noise=polyak_noise,
        random_eps=random_eps,
        history_len=history_len,
        render=render,
    )

    # add to history
    self.add_history_keys(["reward_per_eps"])
    self.add_history_keys(["info_" + x + "_mean" for x in self.info_keys])
    self.add_history_keys(["info_" + x + "_min" for x in self.info_keys])
    self.add_history_keys(["info_" + x + "_max" for x in self.info_keys])

    #
    assert any([num_steps, num_episodes]) and not all([num_steps, num_episodes])
    self.num_steps = num_steps
    self.num_episodes = num_episodes
    self.eps_length = eps_length
    self.done = True
    self.curr_eps_step = 0

    self.env = make_env()

  def _seed(self, seed):
    """ Set seed for environment
        """
    self.env.seed(seed)

  def _generate_rollouts(self, observers=(), random=False):
    """generate `num_steps` rollouts
    """
    # Information to store
    experiences = {
        k: [] for k in ("o", "o_2", "ag", "ag_2", "u", "g", "g_2", "r", "done")
    }
    info_values = {k: [] for k in self.info_keys}

    current_step = 0
    current_episode = 0
    while (not self.num_steps) or current_step < self.num_steps:
      # start a new episode if the last one is done
      if self.done or (self.curr_eps_step == self.eps_length):
        self.done = False
        self.curr_eps_step = 0
        self._clear_noise_history()  # clear noise history for polyak noise
        state = self.env.reset()
        self.o, self.ag, self.g = state["observation"], state[
            "achieved_goal"], state["desired_goal"]

      # get the action for all envs of the current batch
      if random:
        u = self._random_action(1)
      else:
        policy_output = self.policy.get_actions(self.o, self.g)
        u = self._add_noise_to_action(policy_output)
      u = u.reshape(self.dims["u"])  # make sure that the shape is correct
      # compute new states and observations
      try:
        state, r, self.done, iv = self.env.step(u)
        o_2, ag_2 = state["observation"], state["achieved_goal"]
        if self.render:
          self.env.render()
      except MujocoException:
        logger.warn(
            "MujocoException caught during rollout generation. Trying again...")
        self.done = True
        return self.generate_rollouts()
      if np.isnan(o_2).any():
        logger.warn("NaN caught during rollout generation. Trying again...")
        self.done = True
        return self.generate_rollouts()

      for observer in observers:
        observer(
            o=self.o,
            ag=self.ag,
            g=self.g,
            o_2=o_2,
            ag_2=ag_2,
            g_2=self.g,
            u=u,
            r=r,
            done=self.done,
            info=iv,
            reset=self.done or (self.curr_eps_step + 1 == self.eps_length),
        )

      experiences["o"].append(self.o)
      experiences["ag"].append(self.ag)
      experiences["g"].append(self.g)
      experiences["o_2"].append(o_2)
      experiences["ag_2"].append(ag_2)
      experiences["g_2"].append(self.g)
      experiences["u"].append(u)
      experiences["r"].append(r)
      experiences["done"].append(self.done)
      for key in self.info_keys:
        info_values[key].append(iv[key])
      self.o = o_2  # o_2 -> o
      self.ag = ag_2  # ag_2 -> ag

      current_step += 1
      self.curr_eps_step += 1
      if self.done or (self.curr_eps_step == self.eps_length):
        current_episode += 1
        if self.num_episodes and current_episode == self.num_episodes:
          break

    # Store all information into an episode dict
    for key, value in experiences.items():
      experiences[key] = np.array(value).reshape(len(value), -1)
    for key, value in info_values.items():
      experiences["info_" + key] = np.array(value).reshape(len(value), -1)

    # Store stats
    # total reward
    total_reward = np.sum(
        experiences["r"]) / current_episode if self.num_episodes else np.NaN
    self.history["reward_per_eps"].append(total_reward)
    # info values
    for key in self.info_keys:
      self.history["info_" + key + "_mean"].append(np.mean(info_values[key]))
      self.history["info_" + key + "_min"].append(np.min(info_values[key]))
      self.history["info_" + key + "_max"].append(np.max(info_values[key]))
    # number of steps
    self.total_num_episodes += current_episode
    self.total_num_steps += current_step

    return experiences


class EpisodeBasedDriver(StepBasedDriver):

  def __init__(self,
               make_env,
               policy,
               dims,
               eps_length,
               max_u,
               num_steps=None,
               num_episodes=None,
               noise_eps=0.0,
               polyak_noise=0.0,
               random_eps=0.0,
               history_len=300,
               render=False,
               **kwargs):

    assert num_episodes and not num_steps

    super(EpisodeBasedDriver, self).__init__(
        make_env,
        policy,
        dims,
        eps_length,
        max_u,
        num_steps,
        num_episodes,
        noise_eps,
        polyak_noise,
        random_eps,
        history_len,
        render,
    )

  def _generate_rollouts(self, observers=(), random=False):
    experiences = super()._generate_rollouts(observers=observers, random=random)
    assert all([
        v.shape[0] == self.num_episodes * self.eps_length
        for v in experiences.values()
    ])
    experiences = {
        k: v.reshape((self.num_episodes, self.eps_length, v.shape[-1]))
        for k, v in experiences.items()
    }
    return experiences