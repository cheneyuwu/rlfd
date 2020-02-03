import numpy as np

from td3fd.ddpg.ddpg import DDPG

default_params = {
    # config summary and implementation identifier
    "alg": "ddpg-tf",
    "config": "default",
    # environment config
    "env_name": "FetchReach-v1",
    "r_scale": 1.0,  # scale the reward of the environment down
    "r_shift": 0.0,  # shift the reward of the environment up
    "eps_length": 0,  # overwrite the default length of the episode provided in _max_episode_steps
    "env_args": {},  # extra arguments passed to the environment
    "gamma": None,  # reward discount, usually set to be 0.995, 0.95, set to None to select based on max_eps_length
    "fix_T": True,  # whether or not to fix episode length for all rollouts (if false, then use the ring buffer)
    # DDPG config (with TD3 params)
    "ddpg": {
        # training
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
        "norm_clip": 5,  # normalized observations are cropped to this value
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
            "q_filter": True,  # whether or not a Q value filter should be used on the actor outputs
            "prm_loss_weight": 0.001,  # weight corresponding to the primary loss
            "aux_loss_weight": 0.0078,  # weight corresponding to the auxilliary loss (also called the cloning loss)
        },
        "shaping_params": {
            "num_epochs": int(1e3),
            "batch_size": 128,  # batch size for training the potential function (gan and nf)
            "num_ensembles": 2,
            "nf": {
                "num_masked": 2,  # used only when nf_type is set to realnvp
                "num_bijectors": 6,  # number of bijectors in the normalizing flow
                "layer_sizes": [512, 512],  # number of neurons in each hidden layer
                "prm_loss_weight": 1.0,
                "reg_loss_weight": 500.0,
                "potential_weight": 5.0,
            },
            "gan": {
                "layer_sizes": [256, 256],  # number of neurons in each hidden layer (both generator and discriminator)
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
        "history_len": 300,  # make sure that this is same as number of cycles
    },
    "evaluator": {
        "num_episodes": 10,
        "num_steps": None,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": True,
        "history_len": 300,
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

