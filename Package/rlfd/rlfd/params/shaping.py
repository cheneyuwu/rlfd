# yapf: disable
from copy import deepcopy
"""
TODO: this should be in customized files
insert_peg_2d_nf_params["agent"]["shaping_params"]["NFShaping"]["reg_loss_weight"] = 4e2
insert_peg_2d_nf_params["agent"]["shaping_params"]["NFShaping"]["potential_weight"] = 1e1

insert_peg_params["agent"]["shaping_params"]["NFShaping"]["reg_loss_weight"] = 4e2
insert_peg_params["agent"]["shaping_params"]["NFShaping"]["potential_weight"] = 1e1

pick_place_params["agent"]["shaping_params"]["NFShaping"]["reg_loss_weight"] = 4e2
pick_place_params["agent"]["shaping_params"]["NFShaping"]["potential_weight"] = 1e1
pick_place_params["agent"]["shaping_params"]["GANShaping"]["latent_dim"] = 16
pick_place_params["agent"]["shaping_params"]["GANShaping"]["potential_weight"] = 1.0

metaworld_params["agent"]["shaping_params"]["NFShaping"]["reg_loss_weight"] = 4e2
metaworld_params["agent"]["shaping_params"]["NFShaping"]["potential_weight"] = 1e1
metaworld_params["agent"]["shaping_params"]["GANShaping"]["latent_dim"] = 16
metaworld_params["agent"]["shaping_params"]["GANShaping"]["potential_weight"] = 1.0

reach2d_params["agent"]["shaping_params"]["GANShaping"]["layer_sizes"] = [256, 256]
reach2d_params["agent"]["shaping_params"]["GANShaping"]["latent_dim"] = 4
reach2d_params["agent"]["shaping_params"]["GANShaping"]["gp_lambda"] = 0.5
reach2d_params["agent"]["shaping_params"]["GANShaping"]["potential_weight"] = 1.0
"""

# NF Shaping Parameters
nf_params = {
    "shaping_type": "NFShaping",
    "num_epochs": int(1e4),
    "batch_size": 128,
    "num_ensembles": 2,
    "norm_obs": True,
    "norm_eps": 0.01,
    "norm_clip": 5,
    "num_masked": 2,
    "num_bijectors": 4,
    "layer_sizes": [256, 256],
    "prm_loss_weight": 1.0,
    "reg_loss_weight": 200.0,
    "potential_weight": 3.0,
}

# GAN Shaping Parameters
gan_params = {
    "shaping_type": "GANShaping",
    "num_epochs": int(4e3),
    "batch_size": 128,
    "num_ensembles": 2,
    "norm_obs": True,
    "norm_eps": 0.01,
    "norm_clip": 5,
    "layer_sizes": [256, 256, 256],
    "latent_dim": 6,
    "gp_lambda": 0.1,
    "critic_iter": 5,
    "potential_weight": 3.0,
}

# ORL Shaping Parameters
orl_params = {
    "shaping_type": "OfflineRLShaping",
    "num_epochs": int(4e3),
    "batch_size": 128,
    "num_ensembles": 2,
    "norm_obs": True,
    "norm_eps": 0.01,
    "norm_clip": 5,
    "gamma": 0.99,
    "layer_sizes": [256, 256, 256],
    "q_lr": 1e-3,
    "pi_lr": 1e-3,
    "polyak": 0.995,
}