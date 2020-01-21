from copy import deepcopy
from td3fd.td3.params.pickandplacerandinit import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("Torch_GAN_Shaping",)
params_config["demo_strategy"] = "gan"
params_config["seed"] = tuple(range(5))

# tune parameters
# params_config["shaping"]["gan"]["potential_weight"] = 1.0