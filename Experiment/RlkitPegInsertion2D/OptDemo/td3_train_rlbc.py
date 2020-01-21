from copy import deepcopy
from td3fd.rlkit_td3.params.peginhole2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3_BC_1e-4",)
params_config["demo_strategy"] = "bc"
params_config["seed"] = tuple(range(5))
params_config["trainer_kwargs"]["q_filter"] = False
# tune parameters
params_config["trainer_kwargs"]["prm_loss_weight"] = (1e-4,)

params_config_2 = deepcopy(base_params)
params_config_2["config"] = ("TD3_BC_1e-2",)
params_config_2["demo_strategy"] = "bc"
params_config_2["seed"] = tuple(range(5))
params_config_2["trainer_kwargs"]["q_filter"] = False
# tune parameters
params_config_2["trainer_kwargs"]["prm_loss_weight"] = (1e-2,)