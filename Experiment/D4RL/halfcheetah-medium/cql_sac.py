from copy import deepcopy
from rlfd.params.cql import gym_mujoco_params

params_config = deepcopy(gym_mujoco_params)

params_config["config"] = ("CQL",)

params_config["env_name"] = "halfcheetah-medium-v0"
# Offline training
params_config["offline_num_epochs"] = int(1e3)
# No online training with offline data
params_config["agent"]["online_data_strategy"] = "None"
# Tuned values
params_config["seed"] = tuple(range(5))

# Exp 1 CQL offline with SAC online
params_config["config"] = ("CQL_Offline_SAC_Online",)
params_config["agent"]["norm_obs_offline"] = (True,)
params_config["num_epochs"] = (int(1e3),)
params_config["agent"]["pi_lr"] = (3e-5,)
params_config["agent"]["cql_tau"] = (0.0,)
params_config["agent"]["auto_cql_alpha"] = (True, False)
params_config["agent"]["cql_log_alpha"] = (2.0, 1.0)
params_config["seed"] = tuple(range(4))
