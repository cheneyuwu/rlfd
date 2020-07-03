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

  def __init__(self, make_env, policy, history_len, render):
    """
    Rollout worker generates experience by interacting with one or many environments.

    This base class handles the following:

    Args:
        make_env           (function)    - a factory function that creates a new instance of the environment when called
        policy             (object)      - the policy that is used to act
        num_episodes       (int)         - the number of parallel rollouts that should be used
        history_len        (int)         - length of history for statistics smoothing
        render             (bool)        - whether or not to render the rollouts
    """
    self.make_env = make_env
    self.policy = policy
    self.render = render
    self.history_len = history_len
    self.history = {}
    self.total_num_episodes = 0
    self.total_num_steps = 0

  def seed(self, seed):
    """set seed for internal environments"""
    return self._seed(seed)

  def generate_rollouts(self, observers=()):
    return self._generate_rollouts(observers=observers)

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

  @abc.abstractmethod
  def _seed(self, seed):
    """set seed for environment"""

  @abc.abstractmethod
  def _generate_rollouts(self, observers=()):
    """Performs `num_episodes` rollouts for maximum time horizon `eps_length` with the current policy"""


class StepBasedDriver(Driver):

  def __init__(self,
               make_env,
               policy,
               num_steps=None,
               num_episodes=None,
               history_len=300,
               render=False):
    """
    Rollout worker generates experience by interacting with one or many environments.

    Args:
        make_env           (func)        - a factory function that creates a new instance of the environment when called
        policy             (cls)         - the policy that is used to act
        history_len        (int)         - length of history for statistics smoothing
        render             (bool)        - whether or not to render the rollouts
    """
    super().__init__(make_env=make_env,
                     policy=policy,
                     history_len=history_len,
                     render=render)
    # add to history
    self.add_history_keys(["reward_per_eps"])
    #
    self.env = make_env()
    assert any([num_steps, num_episodes]) and not all([num_steps, num_episodes])
    self.num_steps = num_steps
    self.num_episodes = num_episodes
    self.eps_length = self.env.eps_length
    self.done = True
    self.curr_eps_step = 0

  def _seed(self, seed):
    """Set seed for environment
    """
    self.env.seed(seed)

  def _generate_rollouts(self, observers=()):
    """generate `num_steps` rollouts
    """
    # Information to store
    experiences = {
        k: [] for k in ("o", "o_2", "ag", "ag_2", "u", "g", "g_2", "r", "done")
    }

    current_step = 0
    current_episode = 0
    while (not self.num_steps) or current_step < self.num_steps:
      # start a new episode if the last one is done
      if self.done or (self.curr_eps_step == self.eps_length):
        self.done = False
        self.curr_eps_step = 0
        state = self.env.reset()
        self.o, self.ag, self.g = state["observation"], state[
            "achieved_goal"], state["desired_goal"]
      u = self.policy(self.o, self.g)
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
        observer(o=self.o,
                 ag=self.ag,
                 g=self.g,
                 o_2=o_2,
                 ag_2=ag_2,
                 g_2=self.g,
                 u=u,
                 r=r,
                 done=self.done,
                 info=iv,
                 reset=self.done or (self.curr_eps_step + 1 == self.eps_length))

      experiences["o"].append(self.o)
      experiences["ag"].append(self.ag)
      experiences["g"].append(self.g)
      experiences["o_2"].append(o_2)
      experiences["ag_2"].append(ag_2)
      experiences["g_2"].append(self.g)
      experiences["u"].append(u)
      experiences["r"].append(r)
      experiences["done"].append(self.done)
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

    # Store stats
    # total reward
    total_reward = np.sum(
        experiences["r"]) / current_episode if self.num_episodes else np.NaN
    self.history["reward_per_eps"].append(total_reward)
    # number of steps
    self.total_num_episodes += current_episode
    self.total_num_steps += current_step

    return experiences


class EpisodeBasedDriver(StepBasedDriver):

  def __init__(self,
               make_env,
               policy,
               num_steps=None,
               num_episodes=None,
               history_len=300,
               render=False,
               **kwargs):
    assert num_episodes and not num_steps
    super(EpisodeBasedDriver, self).__init__(make_env, policy, num_steps,
                                             num_episodes, history_len, render)

  def _generate_rollouts(self, observers=()):
    experiences = super()._generate_rollouts(observers=observers)
    assert all([
        v.shape[0] == self.num_episodes * self.eps_length
        for v in experiences.values()
    ])
    experiences = {
        k: v.reshape((self.num_episodes, self.eps_length, v.shape[-1]))
        for k, v in experiences.items()
    }
    return experiences