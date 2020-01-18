from copy import deepcopy
from td3fd.ddpg.params.peginhole2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TF_BC",)
params_config["ddpg"]["sample_demo_buffer"] = True
params_config["ddpg"]["demo_strategy"] = "bc"
params_config["ddpg"]["bc_params"]["prm_loss_weight"] = 1e-2  # (1e-4, 1e-2)  # tune this
params_config["seed"] = 0  # tuple(range(4))
