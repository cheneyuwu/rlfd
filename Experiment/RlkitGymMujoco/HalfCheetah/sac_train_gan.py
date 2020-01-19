from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

# td3
params_config = deepcopy(base_params)
params_config["config"] = ("SAC_GAN_Shaping",)
params_config["env_name"] = "HalfCheetah-v3"
params_config["demo_strategy"] = "gan"
params_config["seed"] = tuple(range(5))
