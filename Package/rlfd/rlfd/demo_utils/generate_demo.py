import json
import os
import pickle
import sys
osp = os.path

import numpy as np
import tensorflow as tf

from rlfd import train, logger
from rlfd.utils.util import set_global_seeds

DEFAULT_PARAMS = {
    "fix_T": True,
    "seed": 0,
    "num_steps": None,
    "num_episodes": 40,
    "render": False,
    "filename": "demo_data.npz",
}


def main(policy, root_dir, **kwargs):
  """Generate demo from policy file"""
  logger.configure()

  # Get default params from config and update params.
  param_file = osp.join(root_dir, "demo_config.json")
  if osp.isfile(param_file):
    with open(param_file, "r") as f:
      env_params = json.load(f)
  else:
    logger.warn(
        "WARNING: demo_config.json not found! using the default parameters.")
    env_params = DEFAULT_PARAMS.copy()
    param_file = osp.join(root_dir, "demo_config.json")
    with open(param_file, "w") as f:
      json.dump(env_params, f)

  set_global_seeds(env_params["seed"])

  with open(policy, "rb") as f:
    policy = pickle.load(f)

  filename = env_params.pop("filename")

  params = env_params.copy()
  params["env_name"] = policy.info["env_name"]
  params["r_scale"] = policy.info["r_scale"]
  params["r_shift"] = policy.info["r_shift"]
  params["env_args"] = policy.info["env_args"]
  make_env, _ = train.get_env_constructor_and_config(params=params)
  eval_driver = train.config_driver(make_env=make_env,
                                    policy=policy.eval_policy,
                                    **env_params)

  episode = eval_driver.generate_rollouts()

  for key, val in eval_driver.logs("test"):
    logger.record_tabular(key, np.mean(val))
  logger.dump_tabular()

  os.makedirs(root_dir, exist_ok=True)
  np.savez_compressed(osp.join(root_dir, filename), **episode)  # save the file
  logger.info("Demo file has been stored into {}.".format(filename))