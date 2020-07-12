import abc
import os
import pickle
osp = os.path

import tensorflow as tf

AGENTS = {}


class Agent(tf.Module, metaclass=abc.ABCMeta):
  """Defines the general interface of an agent, registers subclasses and
  provide saving methods.
  """

  @classmethod
  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    AGENTS[cls.__name__] = cls

  def __init__(self):
    super().__init__()

    self._tf_ckpt = tf.train.Checkpoint(agent=self)
    self._tf_ckpt_manager = None
    self._tf_ckpt_dir = None

  @property
  @abc.abstractmethod
  def expl_policy(self):
    """Policy for exploration"""

  @property
  @abc.abstractmethod
  def eval_policy(self):
    """Policy for evaluation"""

  @abc.abstractmethod
  def before_training_hook(self, data_dir=None, env=None):
    """Adds data to the replay buffer and initalizes shaping"""

  @abc.abstractmethod
  def train_offline(self):
    """Performs one step of offline training"""

  @abc.abstractmethod
  def train_online(self):
    """Performs one step of online training"""

  @abc.abstractmethod
  def store_experiences(self, experiences):
    """Stores online experiences"""

  @staticmethod
  def get_default_params(self):
    """Return default parameters as a dictionary"""
    return {}

  def save(self, policy_path, ckpt_path=None):
    """Pickles the current policy."""
    with open(policy_path, "wb") as f:
      pickle.dump(self, f)

    if ckpt_path == None:
      return
    if self._tf_ckpt_manager == None or ckpt_path != self._tf_ckpt_dir:
      self._tf_ckpt_manager = tf.train.CheckpointManager(self._tf_ckpt,
                                                         ckpt_path,
                                                         max_to_keep=1)
    self._tf_ckpt_manager.save()

  def load(self, ckpt_path):
    """Loads parameters from a checkpoint"""
    # for var in tf.train.list_variables(tf.train.latest_checkpoint(ckpt_path)):
    #   print("agent-->", var)
    result = self._tf_ckpt.restore(tf.train.latest_checkpoint(ckpt_path))
    # Some variables may not have been contructed yet, but they should be for
    # plotting only, so no need to worry about.
    result.assert_existing_objects_matched()

  def __getstate__(self):
    """For pickle. Store weights"""

  def __setstate__(self, state):
    """For pickle. Re-instantiate the class, load weights"""