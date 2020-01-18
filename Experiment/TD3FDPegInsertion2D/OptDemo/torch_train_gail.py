from copy import deepcopy
from td3fd.gail.params.gail_fetch_peg_in_hole_2d import params_config as base_params

# gail
params_config_gail = deepcopy(base_params)
params_config_gail["config"] = ("gail",)
params_config_gail["seed"] = tuple(range(2, 4))

