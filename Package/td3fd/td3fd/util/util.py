"""Mostly adopted from OpenAI baselines: https://github.com/openai/baselines
"""
import functools
import importlib
import inspect
import random

import numpy as np


def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(":")
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def set_global_seeds(seed):
    try:
        import tensorflow as tf
        from tfdeterminism import patch

        tf.compat.v1.reset_default_graph()  # should be removed after tf2
        tf.compat.v1.set_random_seed(seed)
        patch()  # deterministic tensorflow, requires tensorflow-determinism
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)
