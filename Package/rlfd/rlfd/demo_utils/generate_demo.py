import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from rlfd import config, logger
from rlfd.utils.util import set_global_seeds

DEFAULT_PARAMS = {
    "seed": 0,
    "fix_T": True,
    "demo": {
        "num_episodes": 40,
        "random_eps": 0.0,
        "noise_eps": 0.0,
        "render": False
    },
    "filename": "demo_data.npz",
}


def main(policy, root_dir, **kwargs):
  """Generate demo from policy file
  """
  assert root_dir is not None, "must provide the directory to store into"
  assert policy is not None, "must provide the policy"

  # Setup
  logger.configure()
  assert logger.get_dir() is not None

  # Get default params from config and update params.
  param_file = os.path.join(root_dir, "demo_config.json")
  if os.path.isfile(param_file):
    with open(param_file, "r") as f:
      params = json.load(f)
  else:
    logger.warn(
        "WARNING: demo_config.json not found! using the default parameters.")
    params = DEFAULT_PARAMS.copy()
    param_file = os.path.join(root_dir, "demo_config.json")
    with open(param_file, "w") as f:
      json.dump(params, f)

  set_global_seeds(params["seed"])

  with open(policy, "rb") as f:
    policy = pickle.load(f)

  # Extract environment construction information
  params["env_name"] = policy.info["env_name"].replace("Dense", "")
  params["r_scale"] = policy.info["r_scale"]
  params["r_shift"] = policy.info["r_shift"]
  params["eps_length"] = policy.info[
      "eps_length"] if policy.info["eps_length"] != 0 else policy.T
  params["env_args"] = policy.info["env_args"]
  params["gamma"] = policy.info["gamma"]
  params = config.add_env_params(params=params)
  demo = config.config_demo(params=params, policy=policy)

  # Generate demonstration data
  episode = demo.generate_rollouts()

  # Log
  for key, val in demo.logs("test"):
    logger.record_tabular(key, np.mean(val))
  logger.dump_tabular()

  # Store
  os.makedirs(root_dir, exist_ok=True)
  file_name = os.path.join(root_dir, params["filename"])
  np.savez_compressed(file_name, **episode)  # save the file
  logger.info("Demo file has been stored into {}.".format(file_name))


if __name__ == "__main__":

  from rlfd.utils.cmd_util import ArgParser

  ap = ArgParser()
  ap.parser.add_argument("--root_dir",
                         help="policy store directory",
                         type=str,
                         default=None)
  ap.parser.add_argument("--policy",
                         help="input policy file for training",
                         type=str,
                         default=None)
  ap.parse(sys.argv)

  main(**ap.get_dict())
