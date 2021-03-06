# yapf: disable
from copy import deepcopy

# Default Parameters
default_params = {
    # config summary
    "algo": "CQL",
    "config": "default",
    # environment config
    "env_name": "InvertedPendulum-v2",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 0,
    "env_args": {},
    "fix_T": False,
    # learner
    "pretrained": None,
    "offline_num_epochs": int(1e3),
    "offline_num_batches_per_epoch": 1000,
    "random_expl_num_cycles": 0,
    "num_epochs": 0,
    "num_cycles_per_epoch": 1,
    "num_batches_per_cycle": 1000,
    "expl_num_episodes_per_cycle": None,
    "expl_num_steps_per_cycle": 1000,
    "eval_num_episodes_per_cycle": 5,
    "eval_num_steps_per_cycle": None,
    # agent config
    "agent": {
        "gamma": 0.99,
        "offline_batch_size": 256,
        "online_batch_size": 256,
        "online_sample_ratio": 1,
        # replay buffer setup
        "buffer_size": int(1e6),
        # online training use a pre-trained actor or critic
        "use_pretrained_actor": False,
        "use_pretrained_critic": False,
        "use_pretrained_alpha": False,
        # online training plus offline data
        "online_data_strategy": "None",  # ["None", "BC", "Shaping"]
        # normalize observation
        "norm_obs_online": True,
        "norm_obs_offline": True,
        "norm_eps": 0.01,
        "norm_clip": 5,
        # actor critic networks
        "layer_sizes": [256, 256],
        "auto_alpha": True,
        "alpha": 0.2,
        # cql specific
        "cql_tau": 0.0,
        "auto_cql_alpha": True,
        "cql_log_alpha": 2.0,
        "cql_alpha_lr": 3e-4,
        "cql_weight_decay_factor": 1,
        #
        "q_lr": 3e-4,
        "pi_lr": 3e-5,
        "action_l2": 0.0,
        # double q learning
        "soft_target_tau": 5e-3,
        "target_update_freq": 1,
    },
    "seed": 0,
}

# OpenAI Gym
gym_mujoco_params = deepcopy(default_params)
