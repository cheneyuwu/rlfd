import abc
import pickle

AGENTS = {}


class Agent(object, metaclass=abc.ABCMeta):

  @classmethod
  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    AGENTS[cls.__name__] = cls

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

  def save(self, path):
    """Pickles the current policy."""
    with open(path, "wb") as f:
      pickle.dump(self, f)

  def __getstate__(self):
    """For pickle. Store weights"""

  def __setstate__(self, state):
    """For pickle. Re-instantiate the class, load weights"""