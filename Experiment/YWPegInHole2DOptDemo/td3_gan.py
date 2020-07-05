from copy import deepcopy
from rlfd.params.td3 import insert_peg_2d_params
from rlfd.params.shaping import gan_params

params_config = deepcopy(insert_peg_2d_params)
gan_params_config = deepcopy(gan_params)

params_config["config"] = ("TD3_GANShaping",)
# GAN Shaping
gan_params_config["num_epochs"] = (int(1e4),)
gan_params_config["gp_lambda"] = (1.0,)
params_config["shaping"] = gan_params_config
# Offline training with bc
params_config["offline_num_epochs"] = 100
# Online training with shaping
params_config["agent"]["online_data_strategy"] = "Shaping"
# Tuned values
params_config["seed"] = tuple(range(5))
