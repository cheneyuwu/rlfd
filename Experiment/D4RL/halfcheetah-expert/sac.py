from copy import deepcopy
from rlfd.params.sac import gym_mujoco_params

params_config = deepcopy(gym_mujoco_params)

params_config["config"] = ("SAC",)

params_config["env_name"] = "halfcheetah-expert-v0"  # "InvertedPendulum-v2"
# No offline training
params_config["offline_num_epochs"] = 0
# No online training with offline data
params_config["agent"]["online_data_strategy"] = "None"
# Tuned values
params_config["seed"] = tuple(range(5))

params_config["agent"]["norm_obs"] = (True, False)
params_config["agent"]["q_lr"] = (1e-4, 3e-4)
params_config["agent"]["pi_lr"] = (1e-4, 3e-4)
params_config["seed"] = tuple(range(4))