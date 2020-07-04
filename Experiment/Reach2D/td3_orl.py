from copy import deepcopy
from rlfd.params.td3 import reach2d_params

params_config = deepcopy(reach2d_params)

params_config["config"] = ("TD3_ORLShaping",)

# Offline training with bc
params_config["offline_num_epochs"] = 100
# Online training with shaping
params_config["agent"]["demo_strategy"] = "OfflineRLShaping"
params_config["agent"]["sample_demo_buffer"] = True
# Tuned values
params_config["seed"] = tuple(range(5))
