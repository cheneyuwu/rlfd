from copy import deepcopy
from td3fd.ddpg.params.reach2d import params_config as base_params

# td3fd via normalizing flow (MAF) shaping
params_config = deepcopy(base_params)
params_config["config"] = ("tf_TD3_NF",)
params_config["ddpg"]["demo_strategy"] = "nf"
params_config["ddpg"]["sample_demo_buffer"] = True
params_config["seed"] = tuple(range(2))