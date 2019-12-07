from copy import deepcopy
from td3fd.td3.params.td3 import params_config as base_params

# TD3
params_config = deepcopy(base_params)
params_config["config"] = "TD3"
params_config["env_name"] = ("InvertedPendulum-v2",)  # try also HalfCheetah-v2
params_config["demo_strategy"] = ("gan",)  # use "none" for td3, "bc" for rl+bc, "gan" for shaping
params_config["seed"] = 1  # try also HalfCheetah-v2
