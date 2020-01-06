import numpy as np

from td3fd.bc.bc import BC

default_params = {
    # config summary and implementation identifier
    "alg": "bc-torch",
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
    # BC config
    "bc": {
        # training
        "num_epochs": 10,
        "batch_size": 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        # actor critic networks
        "layer_sizes": [256, 256, 256],  # number of neurons in each hidden layer
        "pi_lr": 0.001,  # actor learning rate
    },
    "memory": {
        # replay buffer setup
        "buffer_size": int(1e6),
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


def configure_bc(params):
    # Extract relevant parameters.
    bc_params = params["bc"]
    # Update parameters
    bc_params.update(
        {
            "dims": params["dims"].copy(),  # agent takes an input observations
            "max_u": params["max_u"],
            # "gamma": params["gamma"],
            "norm_obs": params["norm_obs"],
            "norm_eps": params["norm_eps"],
            "norm_clip": params["norm_clip"],
            # "demo_strategy": params["demo_strategy"],
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
    policy = BC(**bc_params)
    return policy