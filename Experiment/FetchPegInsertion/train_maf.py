from copy import deepcopy
from td3fd.ddpg.param.ddpg_fetch_peg_in_hole_rand_init import params_config as base_params

# maf
params_config = deepcopy(base_params)
params_config["config"] = ("TD3_MAF_Shaping",)
params_config["ddpg"]["demo_strategy"] = "nf"
params_config["ddpg"]["sample_demo_buffer"] = True
params_config["train"]["shaping_n_epochs"] = int(2e4)
params_config["seed"] = 0