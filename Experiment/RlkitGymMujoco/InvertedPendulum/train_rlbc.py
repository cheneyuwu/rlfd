from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

# td3
params_config = deepcopy(base_params)
params_config["config"] = ("SAC_BC",)
params_config["env_name"] = "InvertedPendulum-v2"
params_config["demo_strategy"] = "bc"
params_config["seed"] = 0
