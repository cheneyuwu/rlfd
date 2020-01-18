from copy import deepcopy
from td3fd.ddpg.params.reach2d import params_config as base_params

# td3fd via GAN shaping
params_config = deepcopy(base_params)
params_config["config"] = ("TF_GAN_Shaping",)
params_config["ddpg"]["demo_strategy"] = "gan"
params_config["ddpg"]["sample_demo_buffer"] = True
params_config["seed"] = 0