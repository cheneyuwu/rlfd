import numpy as np
import tensorflow as tf

from td3fd import logger
from td3fd.env_manager import EnvManager
from td3fd.gail.gail import GAIL
from td3fd.gail.rollout import RolloutWorker, SerialRolloutWorker

default_params = {
    # config summary
    "config": "default",
    # environment config
    "env_name": "FetchPegInHole2DVersion-v1",
    "r_scale": 1.0,  # scale the reward of the environment down
    "r_shift": 0.0,  # shift the reward of the environment up
    "eps_length": 40,  # overwrite the default length of the episode provided in _max_episode_steps
    "env_args": {},  # extra arguments passed to the environment
    "fix_T": True,  # whether or not to fix episode length for all rollouts (if false, then use the ring buffer)
    # GAIL config
    "gail": {
        "num_epochs": int(4e2),
        # replay buffer size (gail and demo)
        "buffer_size": int(1e6),
        # demonstrations
        "num_demo": 40,
        "scope": "gail",
        "policy_step": 3,
        "disc_step": 1,
        "gen_layer_sizes": [100, 100],
        "disc_layer_sizes": [100, 100],
        "max_kl": 0.01,
        "gen_ent_coeff": 0,
        "disc_ent_coeff": 1e-3,
        "lam": 0.97,  # value used by openai for their mujoco environments
        "cg_damping": 3e-4,
        "cg_iters": 10,
        "vf_iters": 5,
        "vf_batch_size": 20,
        # normalizer
        "norm_eps": 0.01,
        "norm_clip": 5,
    },
    # rollouts config
    "rollout": {
        "rollout_batch_size": 1,
        "noise_eps": 0.0,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        "polyak_noise": 0.0,  # use polyak_noise * last_noise + (1 - polyak_noise) * curr_noise
        "random_eps": 0.0,  # percentage of time a random action is taken
        "compute_q": False,
    },
    "evaluator": {
        "rollout_batch_size": 20,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": False,
    },
    "seed": 0,
}


def add_env_params(params):
    env_manager = EnvManager(
        env_name=params["env_name"],
        env_args=params["env_args"],
        r_scale=params["r_scale"],
        r_shift=params["r_shift"],
        eps_length=params["eps_length"],
    )
    params["make_env"] = env_manager.get_env
    tmp_env = params["make_env"]()
    params["eps_length"] = tmp_env.eps_length
    params["gamma"] = 1.0 - 1.0 / params["eps_length"]
    params["max_u"] = np.array(tmp_env.max_u) if isinstance(tmp_env.max_u, list) else tmp_env.max_u
    # get environment dimensions
    tmp_env.reset()
    obs, _, _, info = tmp_env.step(tmp_env.action_space.sample())
    dims = {
        "o": obs["observation"].shape,  # the state
        "g": obs["desired_goal"].shape,  # extra state that does not change within 1 episode
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


def configure_gail(params):
    # Extract relevant parameters.
    gail_params = params["gail"]

    # Update parameters
    gail_params.update(
        {
            "max_u": params["max_u"],
            "input_dims": params["dims"].copy(),  # agent takes an input observations
            "eps_length": params["eps_length"],
            "fix_T": params["fix_T"],
            "gamma": params["gamma"],
            "info": {
                "env_name": params["env_name"],
                "r_scale": params["r_scale"],
                "r_shift": params["r_shift"],
                "eps_length": params["eps_length"],
                "env_args": params["env_args"],
            },
        }
    )
    policy = GAIL(**gail_params)
    return policy


def config_rollout(params, policy):
    rollout_params = params["rollout"]
    rollout_params.update({"dims": params["dims"], "eps_length": params["eps_length"], "max_u": params["max_u"]})
    return _config_rollout_worker(params["make_env"], params["fix_T"], params["seed"], policy, rollout_params)


def config_evaluator(params, policy):
    rollout_params = params["evaluator"]
    rollout_params.update({"dims": params["dims"], "eps_length": params["eps_length"], "max_u": params["max_u"]})
    return _config_rollout_worker(params["make_env"], params["fix_T"], params["seed"], policy, rollout_params)


def config_demo(params, policy):
    rollout_params = params["demo"]
    rollout_params.update({"dims": params["dims"], "eps_length": params["eps_length"], "max_u": params["max_u"]})
    return _config_rollout_worker(params["make_env"], params["fix_T"], params["seed"], policy, rollout_params)


def _config_rollout_worker(make_env, fix_T, seed, policy, rollout_params):

    if fix_T:  # fix the time horizon, so use the parrallel virtual envs
        rollout = RolloutWorker(make_env, policy, **rollout_params)
    else:
        rollout = SerialRolloutWorker(make_env, policy, **rollout_params)
    rollout.seed(seed)

    return rollout
