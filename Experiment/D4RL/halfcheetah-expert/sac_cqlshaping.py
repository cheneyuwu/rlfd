from copy import deepcopy
from rlfd.params.sac import gym_mujoco_params
from rlfd.params.shaping import orl_params

params_config = deepcopy(gym_mujoco_params)
orl_params_config = deepcopy(orl_params)

params_config["config"] = ("SAC_CQLShaping",)

params_config["env_name"] = "halfcheetah-expert-v0"  # "InvertedPendulum-v2"
# ORL Shaping
params_config["shaping"] = orl_params_config
# No offline training
params_config["offline_num_epochs"] = 0
# No online training with offline data
params_config["agent"]["online_data_strategy"] = "Shaping"
# Tuned values
params_config["seed"] = tuple(range(5))

# Exp1: use cql shaping but do not use offline data during online training
# Demo policy is: config_CQL_NoMaxTerm/norm_obs_offline_True/cql_tau_10.0/pi_lr_3e-05/seed_1/policies/offline_policy_initial.pkl`
params_config["agent"]["offline_batch_size"] = (0,)
params_config["agent"]["norm_obs_offline"] = (False,)
params_config["agent"]["norm_obs_online"] = (True,)
params_config["agent"]["q_lr"] = (3e-4,)
params_config["agent"]["pi_lr"] = (3e-4,)
params_config["seed"] = tuple(range(4))