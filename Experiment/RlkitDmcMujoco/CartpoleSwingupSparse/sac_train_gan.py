from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_GAN_Shaping",)
params_config["env_name"] = "cartpole:swingup_sparse"
params_config["demo_strategy"] = "gan"
params_config["seed"] = tuple(range(5))

# tune params
params_config["algorithm_kwargs"]["num_epochs"] = int(2000)
params_config["shaping"]["gan"]["latent_dim"] = 6