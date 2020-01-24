from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_GAN_Shaping",)
params_config["env_name"] = "hammer-v1"
params_config["demo_strategy"] = "gan"
params_config["seed"] = tuple(range(5))

# tune params
params_config["shaping"]["gan"]["latent_dim"] = 10
params_config["shaping"]["gan"]["gp_target"] = 0.1
