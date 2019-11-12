from copy import deepcopy
from td3fd.gail.params.gail_fetch_peg_in_hole_rand_init import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("GAIL",)
params_config["seed"] = 0