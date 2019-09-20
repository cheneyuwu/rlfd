"""Mostly adopted from OpenAI baselines: https://github.com/openai/baselines
"""
import importlib
import inspect
import functools
import numpy as np
import random


def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(":")
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def set_global_seeds(i):
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000000 * rank if i is not None else None
    try:
        import tensorflow as tf

        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)
