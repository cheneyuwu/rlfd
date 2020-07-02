from rlfd import config, learners
from rlfd.sac import sac
from rlfd.sac.params import default_params


def train(root_dir, params):
  config.check_params(params, default_params.parameters)
  learner = learners.Learner(root_dir, sac.SAC, params)
  learner.learn()