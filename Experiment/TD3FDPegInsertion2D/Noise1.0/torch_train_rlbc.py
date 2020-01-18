from copy import deepcopy
from td3fd.td3.params.peginhole2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("Torch_BC",)
params_config["demo_strategy"] = "bc"
params_config["seed"] = 0  # tuple(range(8))

# tune parameters