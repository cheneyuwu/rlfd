from copy import deepcopy
from td3fd.rlkit_sac.params.peginhole2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_GAN_Shaping_GP0.1",)
params_config["demo_strategy"] = "gan"
params_config["seed"] = tuple(range(3))
# tune parameters
params_config["shaping"]["gan"]["lambda_term"] = 1.0
params_config["shaping"]["gan"]["gp_target"] = (0.1,)
