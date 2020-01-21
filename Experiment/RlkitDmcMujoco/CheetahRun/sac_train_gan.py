from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_GAN_Shaping",)
params_config["env_name"] = "cheetah:run"
params_config["demo_strategy"] = "gan"
params_config["seed"] = tuple(range(5))
