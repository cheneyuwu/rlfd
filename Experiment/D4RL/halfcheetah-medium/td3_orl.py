from copy import deepcopy
from rlfd.params.td3 import gym_mujoco_params
from rlfd.params.shaping import orl_params

params_config = deepcopy(gym_mujoco_params)
shaping_params_config = deepcopy(orl_params)

params_config["config"] = ("TD3_OfflineRLShaping",)

params_config["env_name"] = "halfcheetah-medium-v0"
# ORL Shaping
params_config["shaping"] = shaping_params_config
# No offline training
params_config["offline_num_epochs"] = 0
# No online training with offline data
params_config["agent"]["online_data_strategy"] = "Shaping"
# Tuned values
params_config["seed"] = tuple(range(5))

# Exp1
# Uses a critic from offline trained critic via cql: cql_policy.pkl
params_config["seed"] = tuple(range(4))