from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

# td3
params_config = deepcopy(base_params)
params_config["config"] = ("SAC_GAN_Shaping_100.0_weight",)
params_config["env_name"] = "HalfCheetah-v3"
params_config["demo_strategy"] = "gan"
params_config["seed"] = 0

# debug
params_config["shaping"]["gan"]["potential_weight"] = 100.0
params_config["shaping"]["num_epochs"] = int(4e3)