from copy import deepcopy
from td3fd.ddpg.param.ddpg_fetch_peg_in_hole_rand_init import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3_BC",)
params_config["ddpg"]["demo_strategy"] = "bc"
params_config["ddpg"]["sample_demo_buffer"] = True
# the lamda term, try changing it to 1e-1 and see the change in convergence rates
params_config["ddpg"]["bc_params"]["prm_loss_weight"] = (1e-3,)
params_config["seed"] = 0