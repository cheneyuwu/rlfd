from copy import deepcopy
from rlfd.params.td3 import pick_place_params
from rlfd.params.shaping import gan_params

params_config = deepcopy(pick_place_params)
gan_params_config = deepcopy(gan_params)

params_config["config"] = ("TD3_GANShaping",)
# GAN Shaping
gan_params_config["num_epochs"] = (int(1e4),)
gan_params_config["gp_lambda"] = (1.0,)
gan_params_config["latent_dim"] = (16,)
gan_params_config["potential_weight"] = (1.0,)
params_config["shaping"] = gan_params_config
# Offline training with bc
params_config["offline_num_epochs"] = 100
# Online training with shaping
params_config["agent"]["demo_strategy"] = "Shaping"
params_config["agent"]["sample_demo_buffer"] = True
# Tuned values
params_config["seed"] = tuple(range(5))
