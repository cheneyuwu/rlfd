from copy import deepcopy
from rlfd.params.td3 import reach2d_params

params_config = deepcopy(reach2d_params)

params_config["config"] = ("TD3_NFShaping",)
# Offline training with bc
params_config["offline_num_epochs"] = 100
# Online training with shaping
params_config["agent"]["demo_strategy"] = "NFShaping"
params_config["agent"]["sample_demo_buffer"] = True
# Tuned values
params_config["agent"]["shaping_params"]["num_epochs"] = (int(2e3),)
params_config["agent"]["shaping_params"]["NFShaping"]["reg_loss_weight"] = (400.0,)
params_config["agent"]["shaping_params"]["NFShaping"]["potential_weight"] = (5.0,)
params_config["seed"] = tuple(range(2))
