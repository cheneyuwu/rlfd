import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from rlfd import train, logger
from rlfd.utils.util import set_global_seeds

DEFAULT_PARAMS = {
    "fix_T": False,
    "seed": 0,
    "num_steps": None,
    "num_episodes": 10,
    "render": True,
}


def main(policy, **kwargs):
  logger.configure()

  env_params = DEFAULT_PARAMS.copy()

  set_global_seeds(env_params["seed"])

  with open(policy, "rb") as f:
    policy = pickle.load(f)

  params = env_params.copy()
  params["env_name"] = policy.info["env_name"]
  params["r_scale"] = policy.info["r_scale"]
  params["r_shift"] = policy.info["r_shift"]
  params["env_args"] = policy.info["env_args"]
  make_env, _ = train.get_env_constructor_and_config(params=params)
  eval_driver = train.config_driver(make_env=make_env,
                                    policy=policy.eval_policy,
                                    **env_params)

  eval_driver.clear_history()
  eval_driver.generate_rollouts()

  for key, val in eval_driver.logs("test"):
    logger.record_tabular(key, np.mean(val))
  logger.dump_tabular()