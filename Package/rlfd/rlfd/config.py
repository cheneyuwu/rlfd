import numpy as np
import tensorflow as tf

from rlfd.env_manager import EnvManager
from rlfd.drivers import EpisodeBasedDriver, StepBasedDriver
from rlfd.memory import UniformReplayBuffer, RingReplayBuffer


def check_params(params, default_params):
  """make sure that the keys match"""
  assert type(params) == dict
  assert type(default_params) == dict
  for key, value in default_params.items():
    assert key in params.keys(), "missing key: {} in provided params".format(
        key)
    if type(value) == dict:
      check_params(params[key], value)
  for key, value in params.items():
    # assert key in default_params.keys(), "provided params has an extra key: {}".format(key)
    if not key in default_params.keys():
      from termcolor import colored

      print(
          colored("Warning: provided params has an extra key: {}".format(key),
                  "red"))


def add_env_params(params):
  """
    Add the following environment parameters to params:
        make_env, eps_length, gamma, max_u, dims
    """
  env_manager = EnvManager(
      env_name=params["env_name"],
      env_args=params["env_args"],
      r_scale=params["r_scale"],
      r_shift=params["r_shift"],
      eps_length=params["eps_length"],
  )
  params["make_env"] = env_manager.get_env
  tmp_env = params["make_env"]()
  # maximum number of simulation steps per episode
  params["eps_length"] = tmp_env.eps_length
  # calculate discount factor gamma based on episode length
  params["gamma"] = 1.0 - 1.0 / params["eps_length"] if params[
      "gamma"] is None else params["gamma"]
  # limit on the magnitude of actions
  params["max_u"] = np.array(tmp_env.max_u) if isinstance(
      tmp_env.max_u, list) else tmp_env.max_u
  # get environment observation & action dimensions
  tmp_env.reset()
  obs, _, _, info = tmp_env.step(tmp_env.action_space.sample())
  dims = {
      "o": tmp_env.observation_space["observation"].shape,
      "g": tmp_env.observation_space["desired_goal"].shape,
      "u": tmp_env.action_space.shape,
  }
  for key, value in info.items():
    if type(value) == str:
      continue
    value = np.array(value)
    if value.ndim == 0:
      value = value.reshape(1)
    dims["info_{}".format(key)] = value.shape
  params["dims"] = dims
  return params


def config_driver(fix_T, seed, *args, **kwargs):
  driver = (EpisodeBasedDriver(*args, **kwargs) if fix_T else StepBasedDriver(
      *args, **kwargs))
  driver.seed(seed)
  return driver
