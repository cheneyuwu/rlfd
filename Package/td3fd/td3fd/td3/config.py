import numpy as np

from td3fd.td3.ddpg import DDPG

# TODO switch between Shaping for image input and state input. image assumed to be (3, 32, 32)
from td3fd.td3.shaping import GANShaping, NFShaping

# from td3fd.td3.shaping import ImgGANShaping as GANShaping  # This is for image

default_params = {
    # config summary and implementation identifier
    "alg": "ddpg-torch",
    "config": "default",
    # environment config
    "env_name": "FetchReach-v1",
    "r_scale": 1.0,  # scale the reward of the environment down
    "r_shift": 0.0,  # shift the reward of the environment up
    "eps_length": 0,  # overwrite the default length of the episode provided in _max_episode_steps
    "env_args": {},  # extra arguments passed to the environment
    "gamma": None,  # reward discount, usually set to be 0.995, 0.95, set to None to select based on max_eps_length
    "fix_T": True,  # whether or not to fix episode length for all rollouts (if false, then use the ring buffer)
    # normalize observation
    "norm_obs": False,  # whethere or not to normalize observations
    "norm_eps": 0.01,  # epsilon used for observation normalization
    "norm_clip": 5,  # normalized observations are cropped to this value
    # ddpg training
    "num_demo": 40,
    "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
    # DDPG config (with TD3 params)
    "ddpg": {
        # training
        "num_epochs": 10,
        "num_cycles": 10,  # per epoch
        "num_batches": 40,  # training batches per cycle
        "batch_size": 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        "batch_size_demo": 128,  # number of samples to be used from the demonstrations buffer, per mpi thread
        # actor critic networks
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
            "q_filter": False,  # whether or not a Q value filter should be used on the actor outputs
            "prm_loss_weight": 0.001,  # weight corresponding to the primary loss
            "aux_loss_weight": 0.0078,  # weight corresponding to the auxilliary loss (also called the cloning loss)
        },
    },
    # reward shaping
    "shaping": {
        "num_ensembles": 1,
        "num_epochs": int(1e3),
        "batch_size": 64,  # batch size for training the potential function (gan and nf)
        "nf": {
            "num_blocks": 4,
            "num_hidden": 100,
            "prm_loss_weight": 1.0,
            "reg_loss_weight": 200.0,
            "potential_weight": 3.0,
        },
        "gan": {
            "latent_dim": 6,
            "lambda_term": 0.1,
            "layer_sizes": [256, 256],  # number of neurons in each hidden layer (both generator and discriminator)
            "potential_weight": 3.0,
        },
    },
    "memory": {
        # replay buffer setup
        "buffer_size": int(1e6),
    },
    # rollouts config
    "rollout": {
        "num_steps": None,  # number of sim steps per call to generate_rollouts
        "num_episodes": None,  # number of episodes per call to generate_rollouts
        "noise_eps": 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        "polyak_noise": 0.0,  # use polyak_noise * last_noise + (1 - polyak_noise) * curr_noise
        "random_eps": 0.3,  # percentage of time a random action is taken
        "compute_q": False,  # whether or not to compute average q value
        "history_len": 10,  # make sure that this is same as number of cycles
    },
    "evaluator": {
        "num_steps": None,
        "num_episodes": None,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": True,
        "history_len": 1,
    },
    "seed": 0,
}


def configure_ddpg(params):
    # Extract relevant parameters.
    ddpg_params = params["ddpg"]
    # Update parameters
    ddpg_params.update(
        {
            "dims": params["dims"].copy(),  # agent takes an input observations
            "max_u": params["max_u"],
            "gamma": params["gamma"],
            "norm_obs": params["norm_obs"],
            "norm_eps": params["norm_eps"],
            "norm_clip": params["norm_clip"],
            "demo_strategy": params["demo_strategy"],
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


def configure_shaping(params):
    shaping_cls = {"nf": NFShaping, "gan": GANShaping}
    # Extract relevant parameters.
    shaping_params = params["shaping"][params["demo_strategy"]]  # TODO: nf
    # Update parameters
    shaping_params.update(
        {
            "dims": params["dims"].copy(),  # agent takes an input observations
            "max_u": params["max_u"],
            "gamma": params["gamma"],
            "norm_obs": params["norm_obs"],
            "norm_eps": params["norm_eps"],
            "norm_clip": params["norm_clip"],
        }
    )
    policy = shaping_cls[params["demo_strategy"]](**shaping_params)
    return policy
