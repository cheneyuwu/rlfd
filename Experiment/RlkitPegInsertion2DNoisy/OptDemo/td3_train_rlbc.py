from copy import deepcopy
from td3fd.rlkit_td3.params.peginhole2d import params_config as base_params

# td3
params_config = deepcopy(base_params)
params_config["config"] = ("TD3_BC",)
params_config["env_name"] = "YWFetchPegInHole2D-v0"
params_config["demo_strategy"] = "bc"
params_config["seed"] = tuple(range(2))