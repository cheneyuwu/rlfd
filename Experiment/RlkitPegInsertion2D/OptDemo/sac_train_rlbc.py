from copy import deepcopy
from td3fd.rlkit_sac.params.peginhole2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_BC_1e-4",)
params_config["demo_strategy"] = "bc"
params_config["seed"] = tuple(range(2))

# tune parameters
params_config["trainer_kwargs"]["prm_loss_weight"] = (1e-4,)