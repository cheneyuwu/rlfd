from copy import deepcopy
from rlfd.params.cql import gym_mujoco_params

params_config = deepcopy(gym_mujoco_params)

params_config["config"] = ("CQL",)

params_config["env_name"] = "halfcheetah-expert-v0"
# No offline training
params_config["offline_num_epochs"] = int(1e4)
# No online training with offline data
params_config["agent"]["online_data_strategy"] = "None"
# Tuned values
params_config["seed"] = tuple(range(5))