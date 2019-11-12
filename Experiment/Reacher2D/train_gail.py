from copy import deepcopy
from td3fd.gail.params.gail_reacher import params_config as base_params

# GAIL
params_config = deepcopy(base_params)
params_config["config"] = ("GAIL",)
params_config["seed"] = 0

