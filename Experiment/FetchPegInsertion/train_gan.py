from copy import deepcopy
from td3fd.ddpg.param.ddpg_fetch_peg_in_hole_rand_init import params_config as base_params

# TD3 with reward shaping via GAN
params_config = deepcopy(base_params)
params_config["config"] = ("TD3_GAN_Shaping",)
params_config["ddpg"]["demo_strategy"] = "gan"
params_config["ddpg"]["sample_demo_buffer"] = True
params_config["seed"] = 0

