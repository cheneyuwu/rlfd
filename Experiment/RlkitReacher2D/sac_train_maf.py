from copy import deepcopy
from td3fd.rlkit_sac.params.reach2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_MAF_Shaping",)
params_config["demo_strategy"] = "nf"
params_config["seed"] = (0, 1)
