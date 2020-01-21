from copy import deepcopy
from td3fd.rlkit_td3.params.peginholerandinit import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3_GAN_Shaping",)
params_config["demo_strategy"] = "gan"
params_config["seed"] = tuple(range(5))