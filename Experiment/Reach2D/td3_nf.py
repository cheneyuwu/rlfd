from copy import deepcopy
from rlfd.params.td3 import reach2d_params
from rlfd.params.shaping import nf_params

params_config = deepcopy(reach2d_params)
nf_params_config = deepcopy(nf_params)

params_config["config"] = ("TD3_NFShaping",)
# NF Shaping
nf_params_config["num_epochs"] = (int(2e3),)
nf_params_config["reg_loss_weight"] = (400.0,)
nf_params_config["potential_weight"] = (5.0,)
params_config["shaping"] = nf_params_config
# Offline training with bc
params_config["offline_num_epochs"] = 100
# Online training with shaping
params_config["agent"]["online_data_strategy"] = "Shaping"
# Tuned values
params_config["seed"] = tuple(range(5))
