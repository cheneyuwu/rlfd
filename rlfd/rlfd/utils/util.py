"""Mostly adopted from OpenAI baselines: https://github.com/openai/baselines
"""
import functools
import importlib
import inspect
import os
import random

import numpy as np


def set_global_seeds(seed):
  try:
    import tensorflow as tf
    tf.random.set_seed(seed)
  except ImportError:
    print("Warning: tensorflow not installed!")
  try:
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  except ImportError:
    print("Warning: pytorch not installed!")

  np.random.seed(seed)
  random.seed(seed)
