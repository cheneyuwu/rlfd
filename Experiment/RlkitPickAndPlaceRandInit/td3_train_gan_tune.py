from copy import deepcopy
from td3fd.rlkit_td3.params.pickandplacerandinit import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3_GAN_Shaping_Tune",)
params_config["demo_strategy"] = "gan"
params_config["seed"] = tuple(range(1))

# tune
# params_config["shaping"]["num_epochs"] = (100,)
params_config["shaping"]["num_ensembles"] = (4,)
params_config["shaping"]["gan"]["potential_weight"] = (1.0,)
params_config["shaping"]["gan"]["lambda_term"] = (0.2, 0.5, 1.0)
