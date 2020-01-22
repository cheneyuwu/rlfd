from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_BC",)
params_config["env_name"] = "cartpole:swingup_sparse"
params_config["trainer_kwargs"]["q_filter"] = False
params_config["demo_strategy"] = "bc"
params_config["seed"] = tuple(range(5))

# params
params_config["algorithm_kwargs"]["num_epochs"] = int(2000)
