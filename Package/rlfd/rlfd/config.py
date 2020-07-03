import numpy as np
import tensorflow as tf
import termcolor

from rlfd import env_manager, drivers


def check_params(params, default_params):
  """Make sure that the keys match"""
  assert type(params) == dict
  assert type(default_params) == dict
  for key, value in default_params.items():
    if not key in params.keys():
      raise ValueError("missing key: {}".format(key))
    if type(value) == dict:
      check_params(params[key], value)
  for key, value in params.items():
    if not key in default_params.keys():
      print(termcolor.colored("Warning: extra key: {}".format(key), "red"))


def get_env_constructor_and_config(params):
  manager = env_manager.EnvManager(env_name=params["env_name"],
                                   env_args=params["env_args"],
                                   r_scale=params["r_scale"],
                                   r_shift=params["r_shift"])
  tmp_env = manager.get_env()
  tmp_env.reset()
  obs, _, _, _ = tmp_env.step(tmp_env.action_space.sample())
  dims = dict(o=tmp_env.observation_space["observation"].shape,
              g=tmp_env.observation_space["desired_goal"].shape,
              u=tmp_env.action_space.shape)
  info = dict(env_name=params["env_name"],
              env_args=params["env_args"],
              r_scale=params["r_scale"],
              r_shift=params["r_shift"])
  config = dict(eps_length=tmp_env.eps_length,
                fix_T=params["fix_T"],
                max_u=tmp_env.max_u,
                dims=dims,
                info=info)
  return manager.get_env, config


def config_driver(fix_T, seed, *args, **kwargs):
  driver = (drivers.EpisodeBasedDriver(*args, **kwargs)
            if fix_T else drivers.StepBasedDriver(*args, **kwargs))
  driver.seed(seed)
  return driver
