from copy import deepcopy
from rlfd.params.sac import gym_mujoco_params

params_config = deepcopy(gym_mujoco_params)

params_config["config"] = ("SAC",)

params_config["env_name"] = "halfcheetah-medium-v0"
# No offline training
params_config["offline_num_epochs"] = 0
# No online training with offline data
params_config["agent"]["online_data_strategy"] = "None"
# Tuned values
params_config["seed"] = tuple(range(5))

# Exp1
# cql_agent.pkl => pretrained agent via cql
# use cql critic to initialize
# use cql actor to initialize
params_config["config"] = ("SAC_Online_CQLPiInit_CQLQInit",)
params_config["agent"]["use_pretrained_actor"] = True
params_config["agent"]["use_pretrained_critic"] = True
params_config["seed"] = tuple(range(4))