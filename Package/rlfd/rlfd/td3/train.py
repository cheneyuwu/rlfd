from rlfd import config, learners
from rlfd.td3 import td3
from rlfd.td3.params import default_params


def train(root_dir, params):
  config.check_params(params, default_params.parameters)
  learner = learners.Learner(root_dir, td3.TD3, params)
  learner.learn()