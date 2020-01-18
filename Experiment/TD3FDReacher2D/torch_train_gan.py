from copy import deepcopy
from td3fd.td3.params.reach2d import params_config as base_params

# td3fd via GAN shaping
params_config = deepcopy(base_params)
params_config["config"] = ("Torch_GAN_Shaping",)
params_config["demo_strategy"] = "gan"
params_config["seed"] = 0
