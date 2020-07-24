from copy import deepcopy
from rlfd.params.sac import gym_mujoco_params
from rlfd.params.shaping import orl_params

params_config = deepcopy(gym_mujoco_params)
shaping_params_config = deepcopy(orl_params)

params_config["config"] = ("SAC_Online_ORLShaping",)

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
# cql_agent.pkl => pretrained agent via cql
# use cql critic as shaping potential
params_config["config"] = ("SAC_Online_CQLQShaping",)
params_config["seed"] = tuple(range(4))