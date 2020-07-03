from rlfd import config, learners
from . import sac
from .params import default_params


def train(root_dir, params):
  config.check_params(params, default_params.parameters)
  learner = learners.Learner(root_dir, sac.SAC, params)
  learner.learn()