import numpy as np
import tensorflow as tf

from td3fd.env_manager import EnvManager
from yw.ddpg_main.rollout import RolloutWorker, SerialRolloutWorker
from yw.ddpg_main.ddpg import DDPG

default_params = {
    # config summary
    "alg": "old-ddpg-tf",
    "config": "default",
    # environment config
    "env_name": "FetchReach-v1",
    "r_scale": 1.0,  # scale the reward of the environment down
    "r_shift": 0.0,  # shift the reward of the environment up
    "eps_length": 0,  # overwrite the default length of the episode provided in _max_episode_steps
    "env_args": {},  # extra arguments passed to the environment
    "gamma": None,  # the discount rate
    "fix_T": True,  # whether or not to fix episode length for all rollouts (if false, then use the ring buffer)
    # DDPG config
    "ddpg": {
        "num_epochs": 10,
        "num_cycles": 10,  # per epoch
        "num_batches": 40,  # training batches per cycle
        "batch_size": 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        # use demonstrations
        "batch_size_demo": 128,  # number of samples to be used from the demonstrations buffer, per mpi thread
        "sample_demo_buffer": False,  # whether or not to sample from demonstration buffer
        "use_demo_reward": False,  # whether or not to assume that demonstrations have rewards, and train it on the critic
        "num_demo": 0,  # number of expert demo episodes
        "demo_strategy": "none",  # choose between ["none", "bc", "nf", "gan"]
        # normalize observation
        "norm_eps": 0.01,  # epsilon used for observation normalization
        "norm_clip": 5,  # normalized observations are cropped to this values
        # actor critic networks
        "scope": "ddpg",
        "layer_sizes": [256, 256, 256],  # number of neurons in each hidden layer
        "twin_delayed": True,
        "policy_freq": 2,
        "policy_noise": 0.2,
        "policy_noise_clip": 0.5,
        "q_lr": 0.001,  # critic learning rate
        "pi_lr": 0.001,  # actor learning rate
        "action_l2": 1.0,  # quadratic penalty on actions (before rescaling by max_u)
        # double q learning
        "polyak": 0.95,  # polyak averaging coefficient for double q learning
        "bc_params": {
            "q_filter": 1,  # whether or not a Q value filter should be used on the actor outputs
            "prm_loss_weight": 0.001,  # weight corresponding to the primary loss
            "aux_loss_weight": 0.0078,  # weight corresponding to the auxilliary loss (also called the cloning loss)
        },
        "shaping_params": {
            "num_epochs": int(1e4),
            "batch_size": 128,  # batch size for training the potential function (gan and nf)
            "nf": {
                "num_ens": 1,  # number of nf ensembles
                "nf_type": "maf",  # choose between ["maf", "realnvp"]
                "lr": 1e-4,
                "num_masked": 2,  # used only when nf_type is set to realnvp
                "num_bijectors": 6,  # number of bijectors in the normalizing flow
                "layer_sizes": [512, 512],  # number of neurons in each hidden layer
                "initializer_type": "glorot",  # choose between ["zero", "glorot"]
                "prm_loss_weight": 1.0,
                "reg_loss_weight": 500.0,
                "potential_weight": 5.0,
            },
            "gan": {
                "num_ens": 1,  # number of gan ensembles
                "layer_sizes": [256, 256],  # number of neurons in each hidden layer (both generator and discriminator)
                "initializer_type": "glorot",  # choose between ["zero", "glorot"]
                "latent_dim": 6,  # generator latent space dimension
                "gp_lambda": 0.1,  # weight on gradient penalty (refer to WGAN-GP)
                "critic_iter": 5,
                "potential_weight": 3.0,
            },
        },
        # replay buffer setup
        "buffer_size": int(1e6),
    },
    # rollouts config
    "rollout": {
        "num_episodes": 4,
        "num_steps": None,
        "noise_eps": 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        "polyak_noise": 0.0,  # use polyak_noise * last_noise + (1 - polyak_noise) * curr_noise
        "random_eps": 0.3,  # percentage of time a random action is taken
        "compute_q": False,
        "history_len": 10,  # make sure that this is same as number of cycles
    },
    "evaluator": {
        "num_episodes": 10,
        "num_steps": None,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": True,
        "history_len": 1,
    },
    "seed": 0,
}


def check_params(params, default_params=default_params):
    """make sure that the keys match"""
    assert type(params) == dict
    assert type(default_params) == dict
    for key, value in default_params.items():
        assert key in params.keys(), "missing key: {} in provided params".format(key)
        if type(value) == dict:
            check_params(params[key], value)
    for key, value in params.items():
        assert key in default_params.keys(), "provided params has an extra key: {}".format(key)


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
    params["make_env"] = env_manager.get_env
    tmp_env = params["make_env"]()
    # maximum number of simulation steps per episode
    params["eps_length"] = tmp_env.eps_length
    # calculate discount factor gamma based on episode length
    params["gamma"] = 1.0 - 1.0 / params["eps_length"] if params["gamma"] is None else params["gamma"]
    # limit on the magnitude of actions
    params["max_u"] = np.array(tmp_env.max_u) if isinstance(tmp_env.max_u, list) else tmp_env.max_u
    # get environment observation & action dimensions
    tmp_env.reset()
    obs, _, _, info = tmp_env.step(tmp_env.action_space.sample())
    dims = {
        "o": obs["observation"].shape,  # observation
        "g": obs["desired_goal"].shape,  # goal (may be of shape (0,) if not exist)
        "u": tmp_env.action_space.shape,
    }
    for key, value in info.items():
        if type(value) == str: # Note: for now, do not add info str to memory (replay buffer)
            continue
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims["info_{}".format(key)] = value.shape
    params["dims"] = dims
    return params


def configure_ddpg(params):
    # Extract relevant parameters.
    ddpg_params = params["ddpg"] # replay strategy has to be her
    # Update parameters
    ddpg_params.update(
        {
            "dims": params["dims"].copy(),  # agent takes an input observations
            "max_u": params["max_u"],
            "eps_length": params["eps_length"],
            "fix_T": params["fix_T"],
            "gamma": params["gamma"],
            "info": {
                "env_name": params["env_name"],
                "r_scale": params["r_scale"],
                "r_shift": params["r_shift"],
                "eps_length": params["eps_length"],
                "env_args": params["env_args"],
                "gamma": params["gamma"],
            },
        }
    )

    policy = DDPG(**ddpg_params)
    return policy


def config_rollout(params, policy):
    rollout_params = params["rollout"]
    rollout_params.update({"dims": params["dims"], "T": params["eps_length"], "max_u": params["max_u"]})
    return _config_rollout_worker(params["make_env"], params["fix_T"], params["seed"], policy, rollout_params)


def config_evaluator(params, policy):
    rollout_params = params["evaluator"]
    rollout_params.update({"dims": params["dims"], "T": params["eps_length"], "max_u": params["max_u"]})
    return _config_rollout_worker(params["make_env"], params["fix_T"], params["seed"], policy, rollout_params)


def config_demo(params, policy):
    rollout_params = params["demo"]
    rollout_params.update({"dims": params["dims"], "T": params["eps_length"], "max_u": params["max_u"]})
    return _config_rollout_worker(params["make_env"], params["fix_T"], params["seed"], policy, rollout_params)


def _config_rollout_worker(make_env, fix_T, seed, policy, rollout_params):

    if fix_T:  # fix the time horizon, so use the parrallel virtual envs
        rollout = RolloutWorker(make_env, policy, **rollout_params)
    else:
        rollout = SerialRolloutWorker(make_env, policy, **rollout_params)
    rollout.seed(seed)

    return rollout
