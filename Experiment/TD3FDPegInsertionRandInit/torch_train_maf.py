from copy import deepcopy
from td3fd.td3.params.peginholerandinit import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("Torch_MAF_Shaping",)
params_config["demo_strategy"] = "nf"
params_config["seed"] = tuple(range(5))

# tune parameters