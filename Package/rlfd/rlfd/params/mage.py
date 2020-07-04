from copy import deepcopy

# Default Parameters
default_params = {
    # config summary
    "algo": "MAGE",
    "config": "default",
    # environment config
    "env_name": "InvertedPendulum-v2",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 0,
    "env_args": {},
    "fix_T": False,
    # learner
    "offline_num_epochs": 0,
    "offline_num_batches_per_epoch": 0,
    "random_expl_num_cycles": 0,
    "num_epochs": int(1e3),
    "num_cycles_per_epoch": int(1e3),
    "num_batches_per_cycle": 1,
    "expl_num_episodes_per_cycle": None,
    "expl_num_steps_per_cycle": 1,
    "eval_num_episodes_per_cycle": 4,
    "eval_num_steps_per_cycle": None,
    # agent config
    "agent": {
        "gamma": 0.99,
        "online_batch_size": 100,
        "offline_batch_size": 128,
        # replay buffer setup
        "buffer_size": int(1e6),
        # exploration
        "expl_gaussian_noise": 0.1,
        "expl_random_prob": 0.0,
        # use demonstrations
        "sample_demo_buffer": False,
        "use_demo_reward": False,
        "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
        # normalize observation
        "norm_eps": 0.01,
        "norm_clip": 5,
        # actor critic networks
        "layer_sizes": [400, 300],
        "policy_freq": 2,
        "policy_noise": 0.2,
        "policy_noise_clip": 0.5,
        "q_lr": 1e-3,
        "pi_lr": 1e-3,
        "action_l2": 0.0,
        # double q learning
        "polyak": 0.995,
        "target_update_freq": 1,
        # model learning
        "model_update_interval": 250,
        # mage critic loss weight
        "use_model_for_td3_critic_loss": False,
        "critic_loss_weight": 0.01,
        "mage_loss_weight": 1.0,
        "bc_params": {
            "q_filter": False,
            "prm_loss_weight": 1.0,
            "aux_loss_weight": 1.0
        },
        "shaping_params": {
            # potential weight decay
            "potential_decay_scale": 1.0,
            "potential_decay_epoch": 0,
            "num_epochs": int(4e3),
            "batch_size": 128,
            "num_ensembles": 2,
            "nf": {
                "num_masked": 2,
                "num_bijectors": 4,
                "layer_sizes": [256, 256],
                "prm_loss_weight": 1.0,
                "reg_loss_weight": 200.0,
                "potential_weight": 3.0,
            },
            "gan": {
                "layer_sizes": [256, 256, 256],
                "latent_dim": 6,
                "gp_lambda": 0.1,
                "critic_iter": 5,
                "potential_weight": 3.0,
            },
        },
    },
    "seed": 0,
}

# OpenAI Gym
gym_mujoco_params = deepcopy(default_params)
gym_mujoco_params["random_expl_num_cycles"] = int(5e3)

# OpenAI Peg Insertion 2D
insert_peg_2d_params = deepcopy(default_params)
insert_peg_2d_params["env_name"] = "YWFetchPegInHole2D-v0"
insert_peg_2d_params["fix_T"] = True
insert_peg_2d_params["random_expl_num_cycles"] = 10
insert_peg_2d_params["num_epochs"] = int(4e2)
insert_peg_2d_params["num_cycles_per_epoch"] = 10
insert_peg_2d_params["num_batches_per_cycle"] = 40
insert_peg_2d_params["expl_num_episodes_per_cycle"] = 4
insert_peg_2d_params["expl_num_steps_per_cycle"] = None
insert_peg_2d_params["agent"]["gamma"] = 0.975
insert_peg_2d_params["agent"]["online_batch_size"] = 256
insert_peg_2d_params["agent"]["layer_sizes"] = [256, 256]
insert_peg_2d_params["agent"]["polyak"] = 0.95
insert_peg_2d_params["agent"]["target_update_freq"] = 40
insert_peg_2d_params["agent"]["model_update_interval"] = 400
insert_peg_2d_params["agent"]["shaping_params"]["num_epochs"] = int(1e4)
insert_peg_2d_params["agent"]["shaping_params"]["nf"]["reg_loss_weight"] = 4e2
insert_peg_2d_params["agent"]["shaping_params"]["nf"]["potential_weight"] = 1e1

# OpenAI Peg Insertion
insert_peg_params = deepcopy(default_params)
insert_peg_params["env_name"] = "YWFetchPegInHoleRandInit-v0"
insert_peg_params["fix_T"] = True
insert_peg_params["num_epochs"] = int(4e2)
insert_peg_params["num_cycles_per_epoch"] = 10
insert_peg_params["num_batches_per_cycle"] = 40
insert_peg_params["expl_num_episodes_per_cycle"] = 4
insert_peg_params["expl_num_steps_per_cycle"] = None
insert_peg_params["agent"]["gamma"] = 0.975
insert_peg_params["agent"]["online_batch_size"] = 256
insert_peg_params["agent"]["expl_gaussian_noise"] = 0.2
insert_peg_params["agent"]["expl_random_prob"] = 0.2
insert_peg_params["agent"]["layer_sizes"] = [256, 256, 256]
insert_peg_params["agent"]["policy_noise"] = 0.1
insert_peg_params["agent"]["action_l2"] = 0.4
insert_peg_params["agent"]["polyak"] = 0.95
insert_peg_params["agent"]["target_update_freq"] = 40
insert_peg_params["agent"]["model_update_interval"] = 400
insert_peg_params["agent"]["shaping_params"]["num_epochs"] = int(1e4)
insert_peg_params["agent"]["shaping_params"]["nf"]["reg_loss_weight"] = 4e2
insert_peg_params["agent"]["shaping_params"]["nf"]["potential_weight"] = 1e1

# OpenAI Pick Place
pick_place_params = deepcopy(default_params)
pick_place_params["env_name"] = "YWFetchPegInHoleRandInit-v0"
pick_place_params["fix_T"] = True
pick_place_params["num_epochs"] = int(4e2)
pick_place_params["num_cycles_per_epoch"] = 10
pick_place_params["num_batches_per_cycle"] = 40
pick_place_params["expl_num_episodes_per_cycle"] = 4
pick_place_params["expl_num_steps_per_cycle"] = None
pick_place_params["agent"]["gamma"] = 0.975
pick_place_params["agent"]["online_batch_size"] = 256
pick_place_params["agent"]["expl_gaussian_noise"] = 0.2
pick_place_params["agent"]["expl_random_prob"] = 0.2
pick_place_params["agent"]["layer_sizes"] = [256, 256, 256]
pick_place_params["agent"]["policy_noise"] = 0.1
pick_place_params["agent"]["action_l2"] = 0.4
pick_place_params["agent"]["polyak"] = 0.95
pick_place_params["agent"]["target_update_freq"] = 40
pick_place_params["agent"]["shaping_params"]["num_epochs"] = int(1e4)
pick_place_params["agent"]["shaping_params"]["nf"]["reg_loss_weight"] = 4e2
pick_place_params["agent"]["shaping_params"]["nf"]["potential_weight"] = 1e1
pick_place_params["agent"]["shaping_params"]["gan"]["latent_dim"] = 16
pick_place_params["agent"]["shaping_params"]["gan"]["potential_weight"] = 1.0

# Metaworld
metaworld_params = deepcopy(default_params)
metaworld_params["env_name"] = "reach-v1"
metaworld_params["fix_T"] = True
metaworld_params["num_epochs"] = int(1e4)
metaworld_params["num_cycles_per_epoch"] = 10
metaworld_params["num_batches_per_cycle"] = 40
metaworld_params["expl_num_episodes_per_cycle"] = 4
metaworld_params["expl_num_steps_per_cycle"] = None
metaworld_params["agent"]["online_batch_size"] = 256
metaworld_params["agent"]["target_update_freq"] = 40
metaworld_params["agent"]["expl_gaussian_noise"] = 0.2
metaworld_params["agent"]["layer_sizes"] = [256, 256, 256]
metaworld_params["agent"]["polyak"] = 0.95
metaworld_params["agent"]["shaping_params"]["num_epochs"] = int(1e4)
metaworld_params["agent"]["shaping_params"]["nf"]["reg_loss_weight"] = 4e2
metaworld_params["agent"]["shaping_params"]["nf"]["potential_weight"] = 1e1
metaworld_params["agent"]["shaping_params"]["gan"]["latent_dim"] = 16
metaworld_params["agent"]["shaping_params"]["gan"]["potential_weight"] = 1.0

# Reach2D
reach2d_params = deepcopy(default_params)
reach2d_params["env_name"] = "Reach2DF"
reach2d_params["fix_T"] = True
reach2d_params["num_epochs"] = int(1e2)
reach2d_params["num_cycles_per_epoch"] = 10
reach2d_params["num_batches_per_cycle"] = 40
reach2d_params["expl_num_episodes_per_cycle"] = 4
reach2d_params["expl_num_steps_per_cycle"] = None
reach2d_params["agent"]["gamma"] = 0.95
reach2d_params["agent"]["online_batch_size"] = 256
reach2d_params["agent"]["expl_gaussian_noise"] = 0.2
reach2d_params["agent"]["expl_random_prob"] = 0.2
reach2d_params["agent"]["layer_sizes"] = [256, 256]
reach2d_params["agent"]["action_l2"] = 0.4
reach2d_params["agent"]["polyak"] = 0.95
reach2d_params["agent"]["target_update_freq"] = 40
reach2d_params["agent"]["shaping_params"]["num_epochs"] = int(3e3)
reach2d_params["agent"]["shaping_params"]["gan"]["layer_sizes"] = [256, 256]
reach2d_params["agent"]["shaping_params"]["gan"]["latent_dim"] = 4
reach2d_params["agent"]["shaping_params"]["gan"]["gp_lambda"] = 0.5
reach2d_params["agent"]["shaping_params"]["gan"]["potential_weight"] = 1.0