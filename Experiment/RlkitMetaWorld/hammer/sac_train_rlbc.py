from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_BC",)
params_config["env_name"] = "hammer-v1"
params_config["trainer_kwargs"]["q_filter"] = False
params_config["demo_strategy"] = "bc"
params_config["seed"] = tuple(range(5))

# tune params
params_config["trainer_kwargs"]["bc_criterion"] = "logprob"