from copy import deepcopy
from yw.ddpg_main.param.rl_fetch_pick_and_place_rand_init import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3_MAF_Shaping",)
params_config["ddpg"]["demo_strategy"] = "nf"
params_config["ddpg"]["sample_demo_buffer"] = True
params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (400.0,)
params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (10.0,)
params_config["train"]["shaping_n_epochs"] = (int(1e4),)
params_config["seed"] = tuple(range(3,6))