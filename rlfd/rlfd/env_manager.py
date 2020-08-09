import os

import numpy as np
import gym
import gym.wrappers
from gym import spaces

import d4rl
import gym_rlfd


class EnvWrapper:
  """Wrapper of the environment that does the following:
    1. adjust rewards: r = (r + r_shift) / r_scale
    2. modify state to contain: observation, achieved_goal, desired_goal
  """

  def __init__(self, make_env, r_scale, r_shift):
    self.env = make_env()
    self.r_scale = r_scale
    self.r_shift = r_shift
    if "_max_episode_steps" in self.env.__dict__:
      self.eps_length = self.env._max_episode_steps
    elif "max_path_length" in self.env.__dict__:
      self.eps_length = self.env.max_path_length
    else:
      raise RuntimeError("max episode length unknown.")
    if isinstance(self.env, gym.wrappers.TimeLimit):
      self.env = self.env.env
      env_ = self.env
      while isinstance(env_, gym.Wrapper):
        if isinstance(env_, gym.wrappers.TimeLimit):
          raise ValueError("Can remove only top-level TimeLimit gym.Wrapper.")
          env_ = env_.env
    # need the following properties
    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space
    self.max_u = self.action_space.high[0]

  def reset(self, **kwargs):
    state = self.env.reset(**kwargs)
    return self._transform_state(state)

  def get_dataset(self):
    """This is for d4rl environments only"""
    # observations, actions, rewards, terminals
    try:
      dataset = self.env.get_dataset()
      dataset = dict(o=dataset["observations"][:-1],
                     o_2=dataset["observations"][1:],
                     u=dataset["actions"][:-1],
                     r=dataset["rewards"][:-1].reshape((-1, 1)),
                     done=dataset["terminals"][:-1].reshape((-1, 1)))
    except AttributeError:
      dataset = None
    return dataset

  def render(self, **kwargs):
    return self.env.render(**kwargs)

  def seed(self, seed=0):
    return self.env.seed(seed)

  def step(self, action):
    state, r, done, info = self.env.step(action)
    r = (r + self.r_shift) / self.r_scale
    return self._transform_state(state), r, done, info

  def close(self):
    return self.env.close()


class GoalEnvWrapper(EnvWrapper):

  def __init__(self, make_env, r_scale, r_shift):
    super().__init__(make_env, r_scale, r_shift)

    if not type(self.observation_space) == spaces.Dict:
      self.observation_space = spaces.Dict(
          dict(
              observation=self.env.observation_space,
              desired_goal=spaces.Box(-np.inf,
                                      np.inf,
                                      shape=np.empty(0).shape,
                                      dtype="float32"),
              achieved_goal=spaces.Box(-np.inf,
                                       np.inf,
                                       shape=np.empty(0).shape,
                                       dtype="float32"),
          ))

  def _transform_state(self, state):
    """
    modify state to contain: observation, achieved_goal, desired_goal
    """
    if not type(state) == dict:
      state = {
          "observation": state,
          "achieved_goal": np.empty(0),
          "desired_goal": np.empty(0)
      }
    return state

  def get_dataset(self):
    """This is for d4rl environments only"""
    # observations, actions, rewards, terminals
    dataset = super().get_dataset()
    if dataset:
      dataset["g"] = np.empty((dataset["o"].shape[0], 0))
      dataset["g_2"] = np.empty((dataset["o"].shape[0], 0))
      dataset["ag"] = np.empty((dataset["o"].shape[0], 0))
      dataset["ag_2"] = np.empty((dataset["o"].shape[0], 0))
    return dataset


class NoGoalEnvWrapper(EnvWrapper):

  def __init__(self, make_env, r_scale, r_shift):
    super().__init__(make_env, r_scale, r_shift)

    if type(self.observation_space) is spaces.Dict:
      assert len(self.observation_space["desired_goal"].low.shape) == 1
      assert len(self.observation_space["observation"].low.shape) == 1
      shape = (self.observation_space["observation"].high.shape[0] +
               self.observation_space["desired_goal"].high.shape[0],)
      self.observation_space = spaces.Box(-np.inf,
                                          np.inf,
                                          shape=shape,
                                          dtype="float32")

  def _transform_state(self, state):
    """
    modify state to contain: observation, achieved_goal, desired_goal
    """
    if type(state) == dict:
      state = np.concatenate((state["observation"], state["desired_goal"]))
    return state


class EnvManager:

  def __init__(self,
               env_name,
               env_args={},
               r_scale=1,
               r_shift=0.0,
               with_goal=True):
    self.make_env = None
    # OpenAI Gym envs: https://gym.openai.com/envs/#mujoco
    # D4RL envs: https://github.com/rail-berkeley/d4rl/wiki/Tasks
    # Customized envs: YWPickAndPlaceRandInit-v0, YWPegInHoleRandInit-v0
    #                  YWPegInHole2D-v0
    if self.make_env is None and gym is not None:
      try:
        gym.make(env_name, **env_args)
        self.make_env = lambda: gym.make(env_name, **env_args)
      except gym.error.UnregisteredEnv:
        pass

    if self.make_env is None:
      raise NotImplementedError

    # Add extra properties on the environment.
    self.r_scale = r_scale
    self.r_shift = r_shift
    self.with_goal = with_goal

  def get_env(self):
    if self.with_goal:
      return GoalEnvWrapper(self.make_env, self.r_scale, self.r_shift)
    return NoGoalEnvWrapper(self.make_env, self.r_scale, self.r_shift)


if __name__ == "__main__":
  env_manager = EnvManager("pen-cloned-v0")
  env = env_manager.get_env()
  print("Episode length:", env.eps_length)
  print("Demonstration shapes:")
  dataset = env.get_dataset()
  for k, v in dataset.items():
    print(k, v.shape)
  env.seed(0)
  done = True
  while True:
    if done:
      obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
