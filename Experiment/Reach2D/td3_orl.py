from copy import deepcopy
from rlfd.params.td3 import reach2d_params
from rlfd.params.shaping import orl_params

params_config = deepcopy(reach2d_params)
orl_params_config = deepcopy(orl_params)

params_config["config"] = ("TD3_ORLShaping",)
# ORL Shaping
params_config["shaping"] = orl_params_config
# Offline training with bc
params_config["offline_num_epochs"] = 100
# Online training with shaping
params_config["agent"]["demo_strategy"] = "Shaping"
params_config["agent"]["sample_demo_buffer"] = True
# Tuned values
params_config["seed"] = tuple(range(5))
