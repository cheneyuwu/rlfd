from copy import deepcopy
from rlfd.params.cql_dp import gym_mujoco_params

params_config = deepcopy(gym_mujoco_params)

params_config["config"] = ("CQLDP",)

params_config["env_name"] = "halfcheetah-expert-v0"
params_config["agent"]["target_lower_bound"] = -1000
# Offline training
params_config["offline_num_epochs"] = int(1e3)
# No online training with offline data
params_config["agent"]["online_data_strategy"] = "None"
# Tuned values
params_config["seed"] = tuple(range(5))

# # Exp 1 Hyper-parameter tuning with CQL (only offline part.)
# params_config["config"] = ("CQL_NoMaxTerm",)
# params_config["agent"]["norm_obs_offline"] = (True,)
# params_config["agent"]["pi_lr"] = (3e-5,)
# params_config["agent"]["cql_tau"] = (0.0, 10.0)
# params_config["seed"] = tuple(range(4))

# # Exp 2 CQL with online fine-tuning using CQL
# # Note: This is a change in code not reflected in the parameters.
# params_config["config"] = ("CQLDP_Offline_NoMax_CQLDP_Online",)
# params_config["agent"]["norm_obs_offline"] = (True,)
# params_config["num_epochs"] = (int(1e3),)
# params_config["agent"]["pi_lr"] = (3e-5,)
# params_config["agent"]["cql_tau"] = (0.0,)
# params_config["seed"] = tuple(range(4))

# Exp 3 CQL with online fine-tuning using SAC
params_config["config"] = ("CQLDP_Offline_NoMax_SAC_Online",)
params_config["agent"]["norm_obs_offline"] = (True,)
params_config["num_epochs"] = (int(1e3),)
params_config["agent"]["pi_lr"] = (3e-5,)
params_config["agent"]["cql_tau"] = (0.0,)
params_config["seed"] = tuple(range(4))
