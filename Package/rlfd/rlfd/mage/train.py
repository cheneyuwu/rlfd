from rlfd import config, learners
from rlfd.mage import mage
from rlfd.mage.params import default_params


def train(root_dir, params):
  config.check_params(params, default_params.parameters)
  learner = learners.Learner(root_dir, mage.MAGE, params)
  learner.learn()