from copy import deepcopy
from td3fd.rlkit_td3.params.peginhole2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3_MAF_Shaping",)
params_config["demo_strategy"] = "nf"
params_config["seed"] = tuple(range(1))

# tuning params
params_config["shaping"]["nf"]["reg_loss_weight"] = (5e2,)
# params_config["shaping"]["nf"]["prm_loss_weight"] = (1.0,)
# params_config["shaping"]["nf"]["reg_loss_weight"] = (10e1, 5e1, 1e2)
# params_config["shaping"]["nf"]["potential_weight"] = (1e3, 2e3)