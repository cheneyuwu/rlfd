# yapf: disable
from copy import deepcopy

# Default Parameters
default_params = {
    # config summary
    "algo": "MBPO",
    "config": "default",
    # environment config
    "env_name": "InvertedPendulum-v2",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 0,
    "env_args": {},
    "fix_T": False,
    # learner
    "pretrained": None,  # e.g. "./cql" means "./cql.pkl", ".pkl" auto appended.
    "offline_num_epochs": 0,
    "offline_num_batches_per_epoch": 1000,
    "random_expl_num_cycles": int(1),
    "num_epochs": int(1e3),
    "num_cycles_per_epoch": 1,
    "num_batches_per_cycle": 1000,
    "expl_num_episodes_per_cycle": None,
    "expl_num_steps_per_cycle": 1000,
    "eval_num_episodes_per_cycle": 5,
    "eval_num_steps_per_cycle": None,
    # agent config
    "agent": {
        "gamma": 0.99,
        "online_batch_size": 256,
        "offline_batch_size": 0,
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
        "norm_obs_offline": False,
        "norm_eps": 0.01,
        "norm_clip": 5,
        # actor critic networks
        "layer_sizes": [256, 256],
        "auto_alpha": True,
        "alpha": 0.2,
        "q_lr": 3e-4,
        "pi_lr": 3e-4,
        "action_l2": 0.0,
        # double q learning
        "soft_target_tau": 5e-3,
        "target_update_freq": 1,
        "bc_params": {
            "q_filter": False,
            "prm_loss_weight": 1.0,
            "aux_loss_weight": 1.0
        },
        "model_layer_size": [200, 200, 200, 200],
        "model_weight_decay": [0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
        "num_networks": 7,
        "num_elites": 5,
        "model_lr": 0.001, 
        "model_train_freq": 250,
        "rollout_batch_size": 1000,
        "real_ratio": 0.2,
        "rollout_schedule": [20,100,1,1],
    },
    "seed": 0,
}

# OpenAI Gym
gym_mujoco_params = deepcopy(default_params)