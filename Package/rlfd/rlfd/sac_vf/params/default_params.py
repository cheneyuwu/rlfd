from copy import deepcopy
from rlfd.sac.params.default_params import parameters

parameters = deepcopy(parameters)
parameters["algo"] = "sac-vf"