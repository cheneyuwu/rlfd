from copy import deepcopy
from rlfd.params.sac_vf import gym_mujoco_params

params_config = deepcopy(gym_mujoco_params)

params_config["config"] = ("SAC_VF",)

params_config["env_name"] = "HalfCheetah-v2"  # "InvertedPendulum-v2"
# No offline training
params_config["offline_num_epochs"] = 0
# No online training with offline data
params_config["agent"]["online_data_strategy"] = "None"
# Tuned values
params_config["seed"] = tuple(range(5))
