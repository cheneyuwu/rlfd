from copy import deepcopy
from td3fd.rlkit_sac.params.peginholerandinit import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_BC_QFilter_1e-2",)
params_config["demo_strategy"] = "bc"
params_config["trainer_kwargs"]["q_filter"] = True
params_config["seed"] = tuple(range(5))

# tune parameters
params_config["trainer_kwargs"]["prm_loss_weight"] = (1e-2,)
