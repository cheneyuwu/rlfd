import numpy as np
import tensorflow as tf

from td3fd import logger
from td3fd.env_manager import EnvManager
from td3fd.rollout import RolloutWorker, SerialRolloutWorker

# Helper Functions for Configuration
# =====================================


class EnvCache:
    """Only creates a new environment from the provided function if one has not yet already been
    created.

    This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.

    """

    cached_envs = {}

    @staticmethod
    def get_env(make_env):
        if make_env not in EnvCache.cached_envs.keys():
            EnvCache.cached_envs[make_env] = make_env()
        return EnvCache.cached_envs[make_env]


def add_env_params(params):
    """
    Add the following environment parameters to params:
        make_env, eps_length, gamma, max_u, dims
    """
    env_manager = EnvManager(
        env_name=params["env_name"],
        env_args=params["env_args"],
        r_scale=params["r_scale"],
        r_shift=params["r_shift"],
        eps_length=params["eps_length"],
    )
    logger.info(
        "Using environment {} with r scale down by {} shift by {} and max episode {}".format(
            params["env_name"], params["r_scale"], params["r_shift"], params["eps_length"]
        )
    )
    params["make_env"] = env_manager.get_env
    tmp_env = EnvCache.get_env(params["make_env"])
    params["eps_length"] = tmp_env.eps_length
    params["gamma"] = 1.0 - 1.0 / params["eps_length"]
    assert hasattr(tmp_env, "max_u")
    params["max_u"] = np.array(tmp_env.max_u) if isinstance(tmp_env.max_u, list) else tmp_env.max_u
    # get environment dimensions
    tmp_env.reset()
    obs, _, _, info = tmp_env.step(tmp_env.action_space.sample())
    dims = {
        "o": obs["observation"].shape,  # the observation
        "g": obs["desired_goal"].shape,  # extra observation that does not change within 1 episode
        "u": tmp_env.action_space.shape,
    }
    # temporarily put here as we never run multigoal jobs
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims["info_{}".format(key)] = value.shape
    params["dims"] = dims
    return params


def config_rollout(params, policy):
    rollout_params = params["rollout"]
    rollout_params.update({"dims": params["dims"], "eps_length": params["eps_length"], "max_u": params["max_u"]})

    if params["fix_T"]:  # fix the time horizon, so use the parrallel virtual envs
        rollout_worker = RolloutWorker(params["make_env"], policy, **rollout_params)
    else:
        rollout_worker = SerialRolloutWorker(params["make_env"], policy, **rollout_params)
    rollout_worker.seed(params["seed"])

    return rollout_worker


def config_evaluator(params, policy):
    eval_params = params["evaluator"]
    eval_params.update({"dims": params["dims"], "eps_length": params["eps_length"], "max_u": params["max_u"]})

    if params["fix_T"]:  # fix the time horizon, so use the parrallel virtual envs
        evaluator = RolloutWorker(params["make_env"], policy, **eval_params)
    else:
        evaluator = SerialRolloutWorker(params["make_env"], policy, **eval_params)
    evaluator.seed(params["seed"])

    return evaluator


def config_demo(params, policy):
    demo_params = params["demo"]
    demo_params.update({"dims": params["dims"], "eps_length": params["eps_length"], "max_u": params["max_u"]})

    if params["fix_T"]:  # fix the time horizon, so use the parrallel virtual envs
        demo = RolloutWorker(params["make_env"], policy, **demo_params)
    else:
        demo = SerialRolloutWorker(params["make_env"], policy, **demo_params)
    demo.seed(params["seed"])

    return demo
