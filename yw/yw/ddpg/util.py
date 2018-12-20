import os
import subprocess
import sys

import importlib
import inspect
import functools

import tensorflow as tf
import numpy as np

from yw.ddpg import tf_utils as U


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch


def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(":")
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def reshape_for_broadcasting(source, target):
    """Reshapes a tensor (source) to have the correct shape and dtype of the target
    before broadcasting it with MPI.
    """
    dim = len(target.get_shape())
    shape = ([1] * (dim - 1)) + [-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(zip(argspec.args[-len(argspec.defaults) :], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def transitions_in_episode_batch(episode_batch):
    """
    Number of transitions in a given episode batch.
    """
    shape = episode_batch["u"].shape
    return shape[0] * shape[1]


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}
