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

  def __init__(self, init_args):
    super().__init__()

    self._init_args = init_args
    self._saved_model = dict()
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

  def before_offline_hook(self):
    """Optional: before offline training"""
    pass

  def before_online_hook(self):
    """Optional: before online training"""
    pass

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

  def save_model(self, model):
    """Save a tf model when calling agent.save. Only call this function during
    class initialization.
    
    :param model a dictionary mapping model name to model
    """
    self._saved_model.update(model)

  def get_saved_model(self, model):
    return self._saved_model[model]

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
    state = {
        k: v
        for k, v in self._init_args.items()
        if not k in ["self", "__class__"]
    }
    state["tf_model"] = {
        k: v.get_weights() for k, v in self._saved_model.items()
    }
    return state

  def __setstate__(self, state):
    """For pickle. Re-instantiate the class, load weights"""
    tf_model = state.pop("tf_model")
    self.__init__(**state)
    for k, v in tf_model.items():
      self._saved_model[k].set_weights(tf_model[k])