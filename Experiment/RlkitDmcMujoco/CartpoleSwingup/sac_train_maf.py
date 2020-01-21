from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_MAF_Shaping",)
params_config["env_name"] = "cartpole:swingup"
params_config["demo_strategy"] = "nf"
params_config["seed"] = tuple(range(5))

# params
params_config["algorithm_kwargs"]["num_epochs"] = int(1000)