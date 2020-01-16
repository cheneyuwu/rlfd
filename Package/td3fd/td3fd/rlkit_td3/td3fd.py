from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from rlkit.torch.core import np_to_pytorch_batch

class TD3FDTrainer(TorchTrainer):
    """
    Twin Delayed Deep Deterministic policy gradients
    """

    def __init__(
            self,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            target_policy,
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,

            discount=0.99,
            reward_scale=1.0,

            policy_learning_rate=1e-3,
            qf_learning_rate=1e-3,
            policy_and_target_update_period=2,
            tau=0.005,
            qf_criterion=None,
            optimizer_class=optim.Adam,

            demo_strategy="none",
            demo_batch_size=128,
            prm_loss_weight=1.0,
            aux_loss_weight=1.0,
            q_filter=True,
            demo_replay_buffer=None,
            shaping=None,
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf1 = qf1
        self.qf2 = qf2
        self.policy = policy
        self.target_policy = target_policy
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        self.discount = discount
        self.reward_scale = reward_scale

        self.policy_and_target_update_period = policy_and_target_update_period
        self.tau = tau
        self.qf_criterion = qf_criterion

        self.demo_strategy = demo_strategy
        self.demo_batch_size = demo_batch_size
        self.prm_loss_weight = prm_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.q_filter = q_filter
        self.demo_replay_buffer = demo_replay_buffer
        self.shaping = shaping
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_learning_rate,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_learning_rate,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_learning_rate,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        if self.demo_strategy != "none":
            demo_batch = self.demo_replay_buffer.random_batch(self.demo_batch_size) # todo: make this a hyper parameter
            demo_batch = np_to_pytorch_batch(demo_batch)
            demo_rewards = demo_batch['rewards']
            demo_terminals = demo_batch['terminals']
            demo_obs = demo_batch['observations']
            demo_actions = demo_batch['actions']
            demo_next_obs = demo_batch['next_observations']

            comb_obs = torch.cat((obs, demo_obs))
        else:
            comb_obs = obs
        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)
        noise = ptu.randn(next_actions.shape) * self.target_policy_noise
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise

        target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        if self.demo_strategy in ["gan", "nf"]:
            q_target += self.shaping.reward(obs, None, actions, next_obs, None, noisy_next_actions)        
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf2_loss = bellman_errors_2.mean()

        # extra logging
        if self.demo_strategy in ["gan", "nf"]:
            p_pred = self.shaping.potential(obs, None, actions)
            qp1_pred = p_pred + q1_pred
            qp2_pred = p_pred + q2_pred

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy(comb_obs)
            q_output = self.qf1(comb_obs, policy_actions)
            policy_loss = - q_output.mean()

            # extra bc loss on demonstration data
            if self.demo_strategy in ["gan", "nf"]:
                policy_loss += -self.shaping.potential(comb_obs, None, policy_actions).mean()
            if self.demo_strategy == "bc":
                demo_policy_actions = self.policy(demo_obs)
                if self.q_filter:
                    q_demo_actions = torch.min(
                        self.qf1(demo_obs, demo_actions),
                        self.qf2(demo_obs, demo_actions),
                    ).detach()
                    q_new_demo_actions = torch.min(
                        self.qf1(demo_obs, demo_policy_actions),
                        self.qf2(demo_obs, demo_policy_actions),
                    ).detach()
                    masked_bc_loss = torch.where(
                        q_demo_actions > q_new_demo_actions, 
                        (demo_policy_actions - demo_actions) ** 2, 
                        torch.zeros_like(demo_actions)
                    )
                    bc_loss = torch.mean(masked_bc_loss)
                else: 
                    bc_loss = torch.mean((demo_policy_actions - demo_actions) ** 2)
                policy_loss = self.aux_loss_weight * bc_loss + self.prm_loss_weight * policy_loss # TODO


            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = - q_output.mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.demo_strategy in ["gan", "nf"]:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'P Predictions',
                    ptu.get_numpy(p_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'P plus Q1 Predictions',
                    ptu.get_numpy(qp1_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'P plus Q2 Predictions',
                    ptu.get_numpy(qp2_pred),
                ))              
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            trained_policy=self.policy,
            target_policy=self.target_policy,
            shaping=self.shaping,
        )
