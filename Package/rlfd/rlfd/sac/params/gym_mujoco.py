from copy import deepcopy
from .default_params import parameters

params_config = deepcopy(parameters)
params_config["ddpg"]["random_exploration_cycles"] = int(5e3)