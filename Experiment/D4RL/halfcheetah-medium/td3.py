from copy import deepcopy
from rlfd.params.td3 import gym_mujoco_params

params_config = deepcopy(gym_mujoco_params)

params_config["config"] = ("TD3",)

params_config["env_name"] = "halfcheetah-medium-v0"
# No offline training
params_config["offline_num_epochs"] = 0
# No online training with offline data
params_config["agent"]["online_data_strategy"] = "None"
# Tuned values
params_config["seed"] = tuple(range(5))

# Exp 1 TD3
params_config["seed"] = tuple(range(4))