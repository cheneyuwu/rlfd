"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from td3fd import config
from td3fd.env_manager import EnvManager
from td3fd.rlkit_td3.td3fd import TD3FDTrainer
from td3fd.rlkit_sac.shaping import EnsembleRewardShapingWrapper

import os
import numpy as np
import pickle

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def experiment(root_dir, variant):
    # expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    # eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    env_manager = EnvManager(env_name=variant["env_name"], with_goal=False)
    expl_env = NormalizedBoxEnv(env_manager.get_env())
    eval_env = NormalizedBoxEnv(env_manager.get_env())

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size

    # IMPORTANT: Modify path length to be the environment max path length
    variant["algorithm_kwargs"]["max_path_length"] = expl_env.eps_length

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    es = GaussianStrategy(
        action_space=expl_env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        exploration_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    shaping = EnsembleRewardShapingWrapper(
        env=eval_env,
        demo_strategy=variant["demo_strategy"],
        discount=variant["trainer_kwargs"]["discount"],
        **variant["shaping"],
    )
    demo_replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], expl_env,)  # TODO load from demo buffer
    if variant["demo_strategy"] != "none":
        demo_file = os.path.join(root_dir, "demo_data.npz")
        assert os.path.isfile(demo_file), "demonstration training set does not exist"
        demo_data = pickle.load(open(demo_file, "rb"))
        demo_replay_buffer.add_paths(demo_data)
        if variant["demo_strategy"] in ["gan", "nf"]:
            shaping.train(demo_data)
            shaping.evaluate()

    trainer = TD3FDTrainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        shaping=shaping,
        demo_strategy=variant["demo_strategy"],
        demo_replay_buffer=demo_replay_buffer,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def main(root_dir, params):
    config.check_params(params, DEFAULT_PARAMS)
    setup_logger(
        exp_name=params["config"],
        variant=params,
        log_dir=root_dir,
        suppress_std_out=not (MPI is None or MPI.COMM_WORLD.Get_rank() == 0),
    )
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=True)
    experiment(root_dir, params)


DEFAULT_PARAMS = dict(
    # params required by td3fd for logging
    alg="rlkit-td3",
    config="default",
    env_name="HalfCheetah-v3",
    seed=0,
    # rlkit default params

    algorithm_kwargs=dict(
        num_epochs=3000,
        num_train_loops_per_epoch=1,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=256,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        demo_batch_size=128,
        prm_loss_weight=1.0,
        aux_loss_weight=1.0,
        q_filter=True,
    ),
    qf_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    policy_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    replay_buffer_size=int(1E6),
    demo_strategy="maf",
    shaping=dict(
        num_ensembles=1,
        num_epochs=int(3e3),
        batch_size=128,
        norm_obs=True,
        norm_eps=0.01,
        norm_clip=5,
        nf=dict(
            num_blocks=4,
            num_hidden=100,
            prm_loss_weight=1.0,
            reg_loss_weight=200.0,
            potential_weight=500.0,
        ),
        gan=dict(
            latent_dim=6,
            lambda_term=0.1,  
            gp_target=1.0,          
            layer_sizes=[256, 256, 256], 
            potential_weight=3.0,
        ),
    ),
)

# if __name__ == "__main__":
#     variant = dict(
#         algorithm_kwargs=dict(
#             num_epochs=3000,
#             num_eval_steps_per_epoch=5000,
#             num_trains_per_train_loop=1000,
#             num_expl_steps_per_train_loop=1000,
#             min_num_steps_before_training=1000,
#             max_path_length=1000,
#             batch_size=256,
#         ),
#         trainer_kwargs=dict(
#             discount=0.99,
#         ),
#         qf_kwargs=dict(
#             hidden_sizes=[400, 300],
#         ),
#         policy_kwargs=dict(
#             hidden_sizes=[400, 300],
#         ),
#         replay_buffer_size=int(1E6),
#     )
#     # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
#     setup_logger('rlkit-post-refactor-td3-half-cheetah', variant=variant)
#     experiment(variant)
