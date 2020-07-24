from copy import deepcopy
from rlfd.params.cql import gym_mujoco_params

params_config = deepcopy(gym_mujoco_params)

params_config["config"] = ("CQL",)

params_config["env_name"] = "halfcheetah-medium-v0"
# Tuned values
params_config["seed"] = tuple(range(5))

# Exp 1 CQL offline
params_config["config"] = ("CQL_Offline",)
params_config["agent"]["norm_obs_offline"] = (True,)
params_config["seed"] = tuple(range(4))
