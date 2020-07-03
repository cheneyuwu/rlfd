from copy import deepcopy
from .default_params import parameters

params_config = deepcopy(parameters)
params_config["random_expl_num_cycles"] = int(5e3)