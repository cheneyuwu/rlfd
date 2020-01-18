from copy import deepcopy
from td3fd.td3.params.ddpg_reacher import params_config as base_params

# td3fd via normalizing flow (MAF) shaping
params_config = deepcopy(base_params)
params_config["config"] = ("TD3_MAF_Shaping",)
params_config["demo_strategy"] = "nf"
params_config["seed"] = tuple(range(1))
