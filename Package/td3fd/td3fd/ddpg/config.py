import numpy as np
import tensorflow as tf

from td3fd import logger
from td3fd.ddpg.ddpg import DDPG
from td3fd.ddpg.rollout import RolloutWorker, SerialRolloutWorker
from td3fd.env_manager import EnvManager

DEFAULT_PARAMS = {
    # config summary
    "config": "default",
    # environment config
    "env_name": "FetchReach-v1",
    "r_scale": 1.0,  # scale the reward of the environment down
    "r_shift": 0.0,  # shift the reward of the environment up
    "eps_length": 0,  # overwrite the default length of the episode provided in _max_episode_steps
    "env_args": {},  # extra arguments passed to the environment
    "fix_T": True,  # whether or not to fix episode length for all rollouts (if false, then use the ring buffer)
    # DDPG config
    "ddpg": {
        "num_epochs": 10,
        "num_cycles": 10,  # per epoch
        "num_batches": 40,  # training batches per cycle
        # replay buffer setup
        "buffer_size": int(1e6),
        # actor critic networks
        "scope": "ddpg",
        "use_td3": 1,  # whether or not to use td3
        "layer_sizes": [256, 256, 256],  # number of neurons in each hidden layer
        "q_lr": 0.001,  # critic learning rate
        "pi_lr": 0.001,  # actor learning rate
        "action_l2": 1.0,  # quadratic penalty on actions (before rescaling by max_u)
        "batch_size": 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        # double q learning
        "polyak": 0.95,  # polyak averaging coefficient for double q learning
        # use demonstrations
        "sample_demo_buffer": 0,  # whether or not to sample from demonstration buffer
        "batch_size_demo": 128,  # number of samples to be used from the demonstrations buffer, per mpi thread
        "use_demo_reward": 0,  # whether or not to assume that demonstrations have rewards, and train it on the critic
        "num_demo": 0,  # number of expert demo episodes
        "demo_strategy": "none",  # choose between ["none", "pure_bc", "bc", "nf", "gan"]
        "bc_params": {
            "q_filter": 1,  # whether or not a Q value filter should be used on the actor outputs
            "prm_loss_weight": 0.001,  # weight corresponding to the primary loss
            "aux_loss_weight": 0.0078,  # weight corresponding to the auxilliary loss (also called the cloning loss)
        },
        "shaping_params": {
            "batch_size": 128,  # batch size for training the potential function (gan and nf)
            "num_epochs": int(1e3),
            "nf": {
                "num_ens": 1,  # number of nf ensembles
                "nf_type": "maf",  # choose between ["maf", "realnvp"]
                "lr": 1e-4,
                "num_masked": 2,  # used only when nf_type is set to realnvp
                "num_bijectors": 6,  # number of bijectors in the normalizing flow
                "layer_sizes": [512, 512],  # number of neurons in each hidden layer
                "prm_loss_weight": 1.0,
                "reg_loss_weight": 500.0,
                "potential_weight": 5.0,
            },
            "gan": {
                "num_ens": 1,  # number of gan ensembles
                "layer_sizes": [256, 256],  # number of neurons in each hidden layer (both generator and discriminator)
                "latent_dim": 6,  # generator latent space dimension
                "gp_lambda": 0.1,  # weight on gradient penalty (refer to WGAN-GP)
                "critic_iter": 5,
                "potential_weight": 3.0,
            },
        },
        # normalize observation
        "norm_eps": 0.01,  # epsilon used for observation normalization
        "norm_clip": 5,  # normalized observations are cropped to this values
        # i/o clippings
        "clip_obs": 200.0,
        "clip_pos_returns": False,  # whether or not this environment has positive return.
        "clip_return": False,
    },
    # rollouts config
    "rollout": {
        "rollout_batch_size": 4,
        "noise_eps": 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        "polyak_noise": 0.0,  # use polyak_noise * last_noise + (1 - polyak_noise) * curr_noise
        "random_eps": 0.3,  # percentage of time a random action is taken
        "compute_q": False,
        "history_len": 10,  # make sure that this is same as number of cycles
    },
    "evaluator": {
        "rollout_batch_size": 20,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": True,
        "history_len": 1,
    },
    "seed": 0,
}


def check_params(params, default_params=DEFAULT_PARAMS):
    """make sure that the keys match"""
    assert type(params) == dict
    assert type(default_params) == dict
    for key, value in default_params.items():
        assert key in params.keys(), "missing key: {} in provided params".format(key)
        if type(value) == dict:
            check_params(params[key], value)
    for key, value in params.items():
        assert key in default_params.keys(), "provided params has an extra key: {}".format(key)


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
    assert hasattr(tmp_env, "_max_episode_steps")
    params["eps_length"] = tmp_env._max_episode_steps
    params["gamma"] = 1.0 - 1.0 / params["eps_length"]
    assert hasattr(tmp_env, "max_u")
    params["max_u"] = np.array(tmp_env.max_u) if isinstance(tmp_env.max_u, list) else tmp_env.max_u
    # get environment dimensions
    tmp_env.reset()
    obs, _, _, info = tmp_env.step(tmp_env.action_space.sample())
    dims = {
        "o": obs["observation"].shape[0],  # the state
        "g": obs["desired_goal"].shape[0],  # extra state that does not change within 1 episode
        "u": tmp_env.action_space.shape[0],
    }
    # temporarily put here as we never run multigoal jobs
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims["info_{}".format(key)] = value.shape[0]
    params["dims"] = dims
    return params


def configure_ddpg(params):
    # Extract relevant parameters.
    ddpg_params = params["ddpg"]

    # Update parameters
    ddpg_params.update(
        {
            "max_u": params["max_u"],
            "input_dims": params["dims"].copy(),  # agent takes an input observations
            "eps_length": params["eps_length"],
            "fix_T": params["fix_T"],
            "clip_return": (1.0 / (1.0 - params["gamma"])) if params["ddpg"]["clip_return"] else np.inf,
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
    policy = DDPG(**ddpg_params)
    return policy


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
